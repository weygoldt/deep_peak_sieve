import nixio
import numpy as np
# import ruptures as rpt
import scipy.signal as signal
from dash import Input, Output
from IPython import embed
from joblib import Parallel, delayed
from nixio.exceptions import DuplicateName
from rich.progress import track

from . import peak_detection, preprocessing


def callback_save_processing_channels(app):
    @app.callback(
        Output("spinner_save_preprocessing", "children"),
        Input("bt_save_to_disk", "n_clicks"),
        Input("filepath", "data"),
        Input("sw_bandpass_filter", "value"),
        Input("lowcutoff", "value"),
        Input("highcutoff", "value"),
        Input("sw_common_reference", "value"),
        Input("sw_peaks_current_window", "value"),
        Input("n_median", "value"),
        Input("chunksize", "value"),
        Input("exclude_radius", "value"),
        Input("threshold_artefact", "value"),
        Input("sw_notch_filter", "value"),
        Input("notch", "value"),
    )
    def save(
        n_clicks,
        filepath,
        sw_bandpass,
        low,
        high,
        sw_common_ref,
        sw_peak,
        n_value,
        chunksize_input,
        exclude_radius,
        th_artefact,
        sw_notch,
        notch,
    ):
        if n_clicks == 0:
            return
        nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadWrite)

        recording = nix_file.blocks[0].data_arrays["data"]
        channels = np.arange(recording.shape[1])

        section = nix_file.sections["recording"]
        sample_rate = float(section["samplerate"][0])
        block = nix_file.blocks[0]

        processed_array, channel_data_frame = create_empty_nix(
            block, recording, channels
        )

        probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        common_ref = preprocessing.common_ref_recording_channels(recording)

        peaks = []
        for ch in track(channels, description="Saving Channel"):
            sliced_recording = recording[:, ch]
            sliced_recording = preprocessing.preprocessing_current_slice_save_to_disk(
                sliced_recording,
                sample_rate,
                sw_bandpass,
                low,
                high,
                sw_common_ref,
                common_ref,
                sw_notch,
                notch,
            )
            processed_array[:, ch] = sliced_recording

            peaks_array = peak_detection.peaks_channel(
                sliced_recording, ch, n_value, th_artefact
            )
            peaks.extend(peaks_array)

            channel_array = block.create_data_array(
                name=f"peaks_channel_{ch}",
                array_type="peak_channel_array",
                data=peaks_array["spike_index"],
                dtype=nixio.DataType.Int64,
            )

        peaks = np.sort(peaks)
        channel_data_frame.append(peaks)

        peaks_array, exclude_peaks = peak_detection.exclude_peaks_with_distance(
            peaks, probe_frame, exclude_radius
        )

        try:
            del block.data_frames["spike_times_dataframe_processed"]
            channel_data_frame_excluded = block.create_data_frame(
                "spike_times_dataframe_processed",
                "dataframe",
                col_names=["spike_index", "amplitude", "channel"],
                col_dtypes=[
                    nixio.DataType.Int64,
                    nixio.DataType.Int16,
                    nixio.DataType.Int8,
                ],
            )
        except KeyError:
            channel_data_frame_excluded = block.create_data_frame(
                "spike_times_dataframe_processed",
                "dataframe",
                col_names=["spike_index", "amplitude", "channel"],
                col_dtypes=[
                    nixio.DataType.Int64,
                    nixio.DataType.Int16,
                    nixio.DataType.Int8,
                ],
            )
        peaks_array = np.sort(peaks_array)
        channel_data_frame_excluded.append(peaks_array)

        data_arrays = block.data_arrays
        for ch in channels:
            try:
                del data_arrays[f"mean_ampl_channel_{ch}"]
                del data_arrays[f"seperations_channel_{ch}"]
            except KeyError:
                continue

        filterd_ch = mean_filter(channel_data_frame_excluded, 100, channels)
        all_seps = Parallel(n_jobs=-1)(
            delayed(calc_seperations)(data) for data in filterd_ch.values()
        )

        for ch in channels:
            channel_array = block.create_data_array(
                name=f"mean_ampl_channel_{ch}",
                array_type="ampl",
                data=filterd_ch[ch],
            )
            block.create_data_array(
                name=f"seperations_channel_{ch}",
                array_type="ndarray",
                data=all_seps[ch],
            )

        all_seps = [fch for fch in data_arrays if "seperations_channel" in fch.name]
        filterd_ch_all = [fch for fch in data_arrays if fch.type == "ampl"]
        end_rec = recording.shape[0] / sample_rate

        combined_dtype = np.dtype(
            [
                ("spike_index", "<i8"),
                ("amplitude", "<i2"),
                ("channel", "i1"),
                ("segment", "<i1"),
            ]
        )
        peaks_segments = np.zeros(
            channel_data_frame_excluded.shape[0], dtype=combined_dtype
        )

        i = 0
        for ch in np.arange(32):
            peaks_ch = channel_data_frame_excluded.read_rows(
                channel_data_frame_excluded["channel"] == ch
            )
            seperation_index_channel = all_seps[ch][:]
            if seperation_index_channel[-1] == filterd_ch_all[ch].shape[0]:
                seperation_index_channel[-1] = seperation_index_channel[-1] - 1
            spike_times = peaks_ch["spike_index"] / sample_rate
            seps = spike_times[seperation_index_channel]

            if not seps[0] == 0:
                seps = np.insert(seps, 0, 0)
            if int(seps[-1]) == int(end_rec):
                seps = np.delete(seps, -1)
                seps = np.insert(seps, len(seps), end_rec)

            for s, (lower, upper) in enumerate(zip(seps[:-1], seps[1:])):
                peaks_segment = peaks_ch[
                    (peaks_ch["spike_index"] / sample_rate >= lower)
                    & (peaks_ch["spike_index"] / sample_rate < upper)
                ]
                peaks_segments[i : i + peaks_segment.shape[0]]["spike_index"] = (
                    peaks_segment["spike_index"]
                )
                peaks_segments[i : i + peaks_segment.shape[0]]["amplitude"] = (
                    peaks_segment["amplitude"]
                )
                peaks_segments[i : i + peaks_segment.shape[0]]["channel"] = (
                    peaks_segment["channel"]
                )
                peaks_segments[i : i + peaks_segment.shape[0]]["segment"] = (
                    np.zeros(peaks_segment.shape[0]) + s
                )
                i += peaks_segment.shape[0]

        del block.data_frames["spike_times_dataframe_processed"]
        channel_data_frame_excluded = block.create_data_frame(
            "spike_times_dataframe_processed",
            "dataframe",
            col_names=["spike_index", "amplitude", "channel", "segment"],
            col_dtypes=[
                nixio.DataType.Int64,
                nixio.DataType.Int16,
                nixio.DataType.Int8,
                nixio.DataType.Int8,
            ],
        )
        channel_data_frame_excluded.append(np.sort(peaks_segments))

        nix_file.close()


def callback_save_processing(app):
    @app.callback(
        Output("spinner_save_preprocessing", "children"),
        Input("bt_save_to_disk", "n_clicks"),
        Input("filepath", "data"),
        Input("sw_bandpass_filter", "value"),
        Input("lowcutoff", "value"),
        Input("highcutoff", "value"),
        Input("sw_common_reference", "value"),
        Input("sw_peaks_current_window", "value"),
        Input("n_median", "value"),
        Input("chunksize", "value"),
        Input("threshold_artefact", "value"),
    )
    def save(
        n_clicks,
        filepath,
        sw_bandpass,
        low,
        high,
        sw_common_ref,
        sw_peak,
        n_value,
        chunksize_input,
        th_artefact,
    ):
        if n_clicks == 0:
            return
        nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadWrite)

        recording = nix_file.blocks[0].data_arrays["data"]
        channels = np.arange(recording.shape[1])

        section = nix_file.sections["recording"]
        sample_rate = float(section["samplerate"][0])
        block = nix_file.blocks[0]
        data_arrays = block.data_arrays

        processed_array, channel_data_frame = create_empty_nix(
            block, recording, channels
        )

        if chunksize_input:
            chunksize = int(chunksize_input * sample_rate)
        else:
            print("Using a Chunksize of 10")
            chunksize = int(10 * sample_rate)

        overlap = int(5 * sample_rate)
        stepsize = chunksize - overlap
        n_full_chunks = (recording.shape[0] - overlap) // stepsize
        last_chunk = n_full_chunks * stepsize

        common_ref = preprocessing.common_ref_recording_channels(recording)
        recording = recording - common_ref

        i = 1
        for c in np.arange(0, last_chunk, stepsize):
            print(f"Processing Chunk {i}")
            start = c
            stop = c + chunksize
            stop = min(stop, recording.shape[0])

            sliced_recording = recording[start:stop]
            sliced_recording = preprocessing.preprocessing_current_slice(
                sliced_recording,
                sample_rate,
                sw_bandpass,
                low,
                high,
                sw_common_ref=False,
            )
            if c == 0:
                processed_array[start:stop] = sliced_recording
            else:
                processed_array[start + overlap : stop] = sliced_recording[overlap:]
            i += 1

        print(f"Processing Last Chunk {i}")
        last_chunk_index = max(0, recording.shape[0] - last_chunk)
        sliced_recording = recording[-last_chunk_index:]

        sliced_recording = preprocessing.preprocessing_current_slice(
            sliced_recording, sample_rate, sw_bandpass, low, high, sw_common_ref
        )
        processed_array[-last_chunk_index:] = sliced_recording

        #     peaks_array = peak_detection.peaks_current_slice(
        #         sliced_recording, start, channels, n_value,th_artefact
        #     )
        #     sorted_peaks = peaks_array[np.argsort(peaks_array["spike_index"])]
        #     channel_data_frame.append(sorted_peaks)
        #
        #     for ch in channels:
        #         channel_peaks = peaks_array[peaks_array["channel"] == ch]["spike_index"]
        #         if start == 0:
        #             channel_array = block.create_data_array(
        #                 name=f"peaks_channel_{ch}",
        #                 array_type="peak_channel_array",
        #                 data=channel_peaks,
        #             )
        #         else:
        #             channel_array = data_arrays[f"peaks_channel_{ch}"]
        #             channel_array.append(channel_peaks)
        #
        # peaks_array = peak_detection.peaks_current_slice(
        #     sliced_recording, recording.shape[0] - last_chunk, channels, n_value,th_artefact
        # )
        # sorted_peaks = peaks_array[np.argsort(peaks_array["spike_index"])]
        # channel_data_frame.append(sorted_peaks)
        # for ch in channels:
        #     channel_peaks = peaks_array[peaks_array["channel"] == ch]["spike_index"]
        #
        #     channel_array = data_arrays[f"peaks_channel_{ch}"]
        #     channel_array.append(channel_peaks)

        nix_file.close()


def create_empty_nix(block, recording, channels):
    data_arrays = block.data_arrays
    try:
        processed_array = block.create_data_array(
            name="processed_data",
            array_type="neo.data_array",
            shape=recording.shape,
            dtype=nixio.DataType.Int16,
        )
    except DuplicateName:
        del data_arrays["processed_data"]
        processed_array = block.create_data_array(
            name="processed_data",
            array_type="neo.data_array",
            shape=recording.shape,
            dtype=nixio.DataType.Int16,
        )

    for ch in channels:
        try:
            del data_arrays[f"peaks_channel_{ch}"]
        except KeyError:
            continue
    try:
        del block.data_frames["spike_times_dataframe"]
        channel_data_frame = block.create_data_frame(
            "spike_times_dataframe",
            "dataframe",
            col_names=["spike_index", "amplitude", "channel"],
            col_dtypes=[
                nixio.DataType.Int64,
                nixio.DataType.Int16,
                nixio.DataType.Int16,
            ],
        )
    except KeyError:
        channel_data_frame = block.create_data_frame(
            "spike_times_dataframe",
            "dataframe",
            col_names=["spike_index", "amplitude", "channel"],
            col_dtypes=[
                nixio.DataType.Int64,
                nixio.DataType.Int16,
                nixio.DataType.Int16,
            ],
        )

    return processed_array, channel_data_frame


def convolve_channel(amplitudes, kernel):
    if amplitudes.size > 0:
        return signal.convolve(amplitudes, kernel, mode="same")
    else:
        return amplitudes


def mean_filter(spike_frame, window_size, n_channels):
    window = np.ones(window_size) / window_size
    filtered_channels = {}
    res = Parallel(n_jobs=-1)(
        delayed(convolve_channel)(
            spike_frame.read_rows(spike_frame["channel"] == ch)["amplitude"], window
        )
        for ch in n_channels
    )
    for i, ch in enumerate(n_channels):
        filtered_channels[ch] = res[i]
    return filtered_channels


def calc_seperations(ch_data):
    if np.array(ch_data.size) > 200:
        # algo_c = rpt.KernelCPD(kernel="linear", min_size=50).fit(ch_data)
        # res = algo_c.predict(pen=8000)
        
        return [0]
    else:
        return [0]
