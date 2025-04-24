from numba.experimental.structref import define_boxing
import numpy as np
from rich.progress import track
from scipy.signal import find_peaks
from IPython import embed


def peaks_current_slice(
    sliced_recording, current_time_index, channels, n_median, th_artefact
):
    peaks_array = []
    for ch in channels:
        channel_data = sliced_recording[:, ch]
        thresh = (
            np.median(np.abs(channel_data - np.median(channel_data)) / 0.6745)
            * n_median
        )
        peaks, _ = find_peaks(-channel_data, height=thresh)
        if peaks.size > 0:
            amplitude = channel_data[peaks]
            time = peaks + current_time_index
            [
                peaks_array.append((ti, am, c))
                for ti, am, c in zip(
                    time, amplitude, np.zeros_like(time, dtype=np.int16) + ch
                )
            ]
    peaks_array = np.array(
        peaks_array,
        dtype=[
            ("spike_index", "<i8"),
            ("amplitude", "<i2"),
            ("channel", "<i2"),
        ],
    )
    if th_artefact:
        artefact_index = np.where(peaks_array["amplitude"] < th_artefact)[0]
        peaks_array = np.delete(peaks_array, artefact_index)
    return peaks_array


def exclude_peaks_with_distance(peaks, probe_frame, exclude_radius):

    sorted_y = np.argsort(probe_frame["y"])
    x = probe_frame["x"][sorted_y]
    y = probe_frame["y"][sorted_y]
    contact_ids = probe_frame["contact_ids"][sorted_y]
    contact_ids = contact_ids.astype(np.int32) - 1

    points = np.vstack((x, y)).T
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)

    channels = np.unique(peaks["channel"])
    excluded_peaks = []
    merged_peaks = []
    # for ch in track(np.unique(peaks["channel"]), description="Excluding spikes based on spacial proberties"):
    for ch in channels:
        channel = np.where(contact_ids == ch)[0][0]
        channel_of_intrest = contact_ids[
            np.where((distances[channel] < exclude_radius) & (distances[channel] > 0))[
                0
            ]
        ]
        peaks_roi = []
        for ch_roi in np.hstack((channel_of_intrest, ch)):
            peaks_adjesent = peaks[peaks["channel"] == ch_roi]
            peaks_roi.extend(peaks_adjesent)

        peaks_roi = np.sort(peaks_roi)

        chunks_peaks = np.arange(
            peaks_roi["spike_index"][0], peaks_roi["spike_index"][-1], 100_000
        )

        for start, stop in zip(chunks_peaks[:-1], chunks_peaks[1:]):

            start_idx = np.searchsorted(peaks_roi["spike_index"], start, side="left")
            end_idx = np.searchsorted(peaks_roi["spike_index"], stop, side="right")
            end_idx_new_window = np.searchsorted(
                peaks_roi["spike_index"],
                min(stop + 100, peaks_roi["spike_index"][-1]),
                side="right",
            )
            peaks_new_window = np.array([])
            if not end_idx == end_idx_new_window:
                peaks_new_window = peaks_roi[end_idx:end_idx_new_window]

            peaks_window = peaks_roi[start_idx:end_idx]

            channel_peaks = peaks_window[peaks_window["channel"] == ch]
            not_channel_peaks = peaks_window[peaks_window["channel"] != ch]

            spike_diffs = np.abs(
                channel_peaks["spike_index"][:, np.newaxis]
                - not_channel_peaks["spike_index"]
            )

            close_spikes_row, close_spikes_col = np.where(
                (spike_diffs <= 10) & (spike_diffs >= 0)
            )
            if not close_spikes_row.size > 0 or not close_spikes_col.size > 0:
                continue

            close_sp, count = np.unique(close_spikes_row, return_counts=True)

            merged_temp_peaks = []
            for gr in np.unique(count):
                mult_close = close_sp[np.where(count == gr)[0]]
                mult_close_spikes_index = np.vstack(
                    [np.where(i == close_spikes_row)[0] for i in mult_close]
                )
                peaks_close = np.hstack(
                    (
                        channel_peaks[close_spikes_row[mult_close_spikes_index]][
                            :, 0
                        ].reshape(mult_close_spikes_index.shape[0], 1),
                        not_channel_peaks[close_spikes_col[mult_close_spikes_index]],
                    ),
                )
                peaks_sorted = np.argsort(peaks_close["amplitude"])
                exclude_these_peaks = np.take_along_axis(
                    peaks_close, peaks_sorted, axis=1
                )[:, 1:]
                merge_these_peaks = np.take_along_axis(
                    peaks_close, peaks_sorted, axis=1
                )[:, 0].reshape(mult_close_spikes_index.shape[0], 1)

                won_comparison = np.where(merge_these_peaks["channel"] == ch)[0]

                if peaks_new_window.size > 0:
                    peaks_new_window_diff = np.abs(
                        merge_these_peaks[won_comparison].flatten()["spike_index"][
                            :, np.newaxis
                        ]
                        - peaks_new_window["spike_index"]
                    )
                    index_row, index_col = np.where(peaks_new_window_diff < 10)
                    if not index_row.size > 0 or not index_col.size > 0:
                        pass

                    stack_new = np.hstack(
                        (
                            merge_these_peaks[won_comparison][index_row, :],
                            peaks_new_window[:, np.newaxis][index_col,:],
                        )
                    )

                    sort_new = np.argsort(stack_new, axis=1)
                    merge_new = np.take_along_axis(stack_new, sort_new, axis=1)[
                        :, 0
                    ]
                    won_comparison_new = np.where(merge_new["channel"] == ch)[0]
                    if won_comparison_new.size > 0:
                        if np.all(
                            merge_these_peaks[won_comparison][index_row][
                                won_comparison_new
                            ]
                            == merge_new[won_comparison_new]
                        ):
                            pass
                        else:
                            excluded_peaks.extend(
                                merge_these_peaks[won_comparison][index_row][
                                    won_comparison_new
                                ]
                            )

                if won_comparison.size > 0:
                    exclude_these_peaks = exclude_these_peaks[
                        won_comparison, :
                    ].flatten()
                    merge_these_peaks = merge_these_peaks[won_comparison, :].flatten()

                    excluded_peaks.extend(exclude_these_peaks)
                    merged_temp_peaks.extend(merge_these_peaks)

    excluded_peaks = np.unique(excluded_peaks)
    excluded_peaks = np.sort(excluded_peaks)

    mask = ~np.isin(peaks, excluded_peaks)
    print(
        f"Excluding {np.sum(np.isin(peaks, excluded_peaks))} based on spacial constrains"
    )
    peaks_array = np.sort(peaks[mask])

    exclude_peaks_ref = exclude_based_refactory_disk(peaks_array)
    mask = ~np.isin(peaks_array, exclude_peaks_ref)
    print(
        f"Excluding {np.sum(np.isin(peaks_array, exclude_peaks_ref))} based on ref period constrains"
    )
    peaks_array = np.sort(peaks_array[mask])
    if exclude_peaks_ref.size > 0:
        exclude_all = np.append(excluded_peaks, exclude_peaks_ref)
    else:
        exclude_all = excluded_peaks

    return peaks_array, exclude_all


def exclude_peaks_with_distance_traces(peaks, probe_frame, exclude_radius):
    sorted_y = np.argsort(probe_frame["y"])
    x = probe_frame["x"][sorted_y]
    y = probe_frame["y"][sorted_y]
    contact_ids = probe_frame["contact_ids"][sorted_y]
    contact_ids = contact_ids.astype(np.int32) - 1

    points = np.vstack((x, y)).T
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)

    excluded_peaks = []

    for ch in np.unique(peaks["channel"]):
        channel = np.where(contact_ids == ch)[0][0]
        channel_of_intrest = contact_ids[
            np.where((distances[channel] < exclude_radius) & (distances[channel] > 0))[
                0
            ]
        ]
        peaks_roi = []
        for ch_roi in np.hstack((channel_of_intrest, ch)):
            peaks_adjesent = peaks[peaks["channel"] == ch_roi]
            peaks_roi.extend(peaks_adjesent)

        peaks_roi = np.sort(peaks_roi)
        channel_peaks = peaks_roi[peaks_roi["channel"] == ch]
        not_channel_peaks = peaks_roi[peaks_roi["channel"] != ch]

        spike_diffs = np.abs(
            channel_peaks["spike_index"][:, np.newaxis]
            - not_channel_peaks["spike_index"]
        )

        close_spikes_row, close_spikes_col = np.where(
            (spike_diffs <= 10) & (spike_diffs >= 0)
        )
        if not close_spikes_row.size > 0 or not close_spikes_col.size > 0:
            continue

        close_sp, count = np.unique(close_spikes_row, return_counts=True)

        for gr in np.unique(count):
            mult_close = close_sp[np.where(count == gr)[0]]
            mult_close_spikes_index = np.vstack(
                [np.where(i == close_spikes_row)[0] for i in mult_close]
            )
            peaks_close = np.hstack(
                (
                    channel_peaks[close_spikes_row[mult_close_spikes_index]][
                        :, 0
                    ].reshape(mult_close_spikes_index.shape[0], 1),
                    not_channel_peaks[close_spikes_col[mult_close_spikes_index]],
                ),
            )
            peaks_sorted = np.argsort(peaks_close["amplitude"])
            exclude_these_peaks = np.take_along_axis(peaks_close, peaks_sorted, axis=1)[
                :, 1:
            ]
            merge_these_peaks = np.take_along_axis(peaks_close, peaks_sorted, axis=1)[
                :, 0
            ].reshape(mult_close_spikes_index.shape[0], 1)

            won_comparison = np.where(merge_these_peaks["channel"] == ch)[0]

            if won_comparison.size > 0:
                exclude_these_peaks = exclude_these_peaks[won_comparison, :].flatten()
                excluded_peaks.extend(exclude_these_peaks)

    excluded_peaks = np.unique(excluded_peaks)
    excluded_peaks = np.sort(excluded_peaks)

    mask = ~np.isin(peaks, excluded_peaks)
    peaks_array = np.sort(peaks[mask])

    exclude_peaks_ref = exclude_based_refactory(peaks_array)
    mask = ~np.isin(peaks_array, exclude_peaks_ref)
    peaks_array = np.sort(peaks_array[mask])
    if exclude_peaks_ref.size > 0:
        exclude_all = np.append(excluded_peaks, exclude_peaks_ref)
    else:
        exclude_all = excluded_peaks

    return peaks_array, exclude_all


def exclude_based_refactory_disk(peaks):
    exclude_peaks = []
    for ch in track(np.unique(peaks["channel"]), description="Excluding Spikes based on refactory Period"):
        channel_peaks = peaks[peaks["channel"] == ch]
        chunks_peaks = np.arange(
            channel_peaks["spike_index"][0], channel_peaks["spike_index"][-1], 100_000
        )

        for start, stop in zip(chunks_peaks[:-1], chunks_peaks[1:]):
            start_idx = np.searchsorted(
                channel_peaks["spike_index"], start, side="left"
            )
            end_idx = np.searchsorted(channel_peaks["spike_index"], stop, side="right")
            end_idx_new_window = np.searchsorted(
                channel_peaks["spike_index"],
                min(stop + 100, channel_peaks["spike_index"][-1]),
                side="right",
            )
            peaks_new_window = np.array([])
            if not end_idx == end_idx_new_window:
                peaks_new_window = channel_peaks[end_idx:end_idx_new_window]

            peaks_window = channel_peaks[start_idx:end_idx]

            spike_diffs = np.abs(
                peaks_window["spike_index"][:, np.newaxis] - peaks_window["spike_index"]
            )

            diagonal_mask = np.tril_indices_from(spike_diffs)
            spike_diffs[diagonal_mask] = -1
            close_spikes_row, close_spikes_col = np.where(
                (spike_diffs <= 10) & (spike_diffs > 0)
            )
            if not close_spikes_row.size > 0 or not close_spikes_col.size > 0:
                continue

            close_sp, count = np.unique(close_spikes_row, return_counts=True)

            for gr in np.unique(count):
                mult_close = close_sp[np.where(count == gr)[0]]
                mult_close_spikes_index = np.vstack(
                    [np.where(i == close_spikes_row)[0] for i in mult_close]
                )
                peaks_close = np.hstack(
                    (
                        peaks_window[close_spikes_row[mult_close_spikes_index]][
                            :, 0
                        ].reshape(mult_close_spikes_index.shape[0], 1),
                        peaks_window[close_spikes_col[mult_close_spikes_index]],
                    ),
                )
                peaks_sorted = np.argsort(peaks_close["amplitude"])

                exclude_these_peaks = np.take_along_axis(
                    peaks_close, peaks_sorted, axis=1
                )[:, 1:].flatten()
                exclude_peaks.extend(exclude_these_peaks)

            if peaks_new_window.size > 0:
                diffs_new = np.abs(
                    peaks_window["spike_index"][:, np.newaxis]
                    - peaks_new_window["spike_index"]
                )
                close_spikes_row, close_spikes_col = np.where(
                    (diffs_new <= 10) & (diffs_new > 0)
                )
                if not close_spikes_row.size > 0 and not close_spikes_col.size > 0:
                    continue
                peaks_close = np.hstack(
                    (
                        peaks_window[close_spikes_row][:, np.newaxis],
                        peaks_new_window[close_spikes_col][:, np.newaxis],
                    ),
                )
                peaks_sorted = np.argsort(peaks_close["amplitude"])

                exclude_these_peaks = np.take_along_axis(
                    peaks_close, peaks_sorted, axis=1
                )[:, 1:].flatten()
                exclude_peaks.extend(exclude_these_peaks)

    return np.sort(exclude_peaks)


def exclude_based_refactory(peaks):
    exclude_peaks = []
    for ch in np.unique(peaks["channel"]):

        channel_peaks = peaks[peaks["channel"] == ch]
        spike_diffs = np.abs(
            channel_peaks["spike_index"][:, np.newaxis] - channel_peaks["spike_index"]
        )

        diagonal_mask = np.tril_indices_from(spike_diffs)
        spike_diffs[diagonal_mask] = -1
        close_spikes_row, close_spikes_col = np.where(
            (spike_diffs <= 10) & (spike_diffs > 0)
        )
        if not close_spikes_row.size > 0 or not close_spikes_col.size > 0:
            continue

        close_sp, count = np.unique(close_spikes_row, return_counts=True)

        for gr in np.unique(count):
            mult_close = close_sp[np.where(count == gr)[0]]
            mult_close_spikes_index = np.vstack(
                [np.where(i == close_spikes_row)[0] for i in mult_close]
            )
            peaks_close = np.hstack(
                (
                    channel_peaks[close_spikes_row[mult_close_spikes_index]][
                        :, 0
                    ].reshape(mult_close_spikes_index.shape[0], 1),
                    channel_peaks[close_spikes_col[mult_close_spikes_index]],
                ),
            )
            peaks_sorted = np.argsort(peaks_close["amplitude"])
            exclude_these_peaks = np.take_along_axis(peaks_close, peaks_sorted, axis=1)[
                :, 1:
            ].flatten()
            exclude_peaks.extend(exclude_these_peaks)
    return np.sort(exclude_peaks)


def peaks_channel(channel_data, ch, n_median, th_artefact):
    thresh = (
        np.median(np.abs(channel_data - np.median(channel_data)) / 0.6745) * n_median
    )
    peaks, _ = find_peaks(-channel_data, height=thresh)
    peaks_array = np.zeros_like(
        peaks,
        dtype=[
            ("spike_index", "<i8"),
            ("amplitude", "<i2"),
            ("channel", "<i2"),
        ],
    )
    if peaks.size > 0:
        amplitude = channel_data[peaks]
        channels = np.full_like(peaks, ch, dtype=np.int16)
        peaks_array["spike_index"] = peaks
        peaks_array["amplitude"] = amplitude
        peaks_array["channel"] = channels

    if th_artefact:
        artefact_index = np.where(peaks_array["amplitude"] < th_artefact)[0]
        peaks_array = np.delete(peaks_array, artefact_index)
    return peaks_array
