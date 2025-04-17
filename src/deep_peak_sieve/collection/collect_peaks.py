from typing_extensions import Annotated
from typing import Optional
from pathlib import Path
import numpy as np
from audioio.audioloader import AudioLoader
import humanize
from datetime import timedelta
import typer
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from IPython import embed

from deep_peak_sieve.utils.loggers import get_logger, get_progress, configure_logging
from deep_peak_sieve.collection.filters import bandpass_filter
from deep_peak_sieve.utils.datasets import load_raw_data, save_numpy

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)


def pretty_duration_humanize(seconds: float) -> str:
    """
    Return a human-friendly string of a duration given in seconds.
    """
    return humanize.naturaldelta(timedelta(seconds=seconds))


def initialize_dataset(path: Path):
    """
    Prepare an empty structure into which extracted peaks and metadata will be placed.
    """
    log.debug("Creating empty dataset to save extracted peaks")
    dataset = {
        "peaks": [],
        "channels": [],
        "amplitudes": [],
        "centers": [],
        "start_stop_index": [],
        "rate": None,
    }
    return dataset


def apply_filter(block: np.ndarray, sample_rate: float, params: dict) -> np.ndarray:
    """
    Apply the specified filtering method to the audio block data.
    """
    if params["mode"] == "savgol":
        log.debug("Applying Savitzky-Golay filter")
        return savgol_filter(
            block.T,
            window_length=params["window_length"],
            polyorder=params["polyorder"],
            # params["window_length"],
            # params["polyorder"],
        ).T
    elif params["mode"] == "bandpass":
        log.debug("Applying bandpass filter")
        return bandpass_filter(
            block.T,
            lowcut=params["lowcut"],
            highcut=params["highcut"],
            sample_rate=sample_rate,
        ).T
    elif params["mode"] == "none":
        log.debug("No filtering applied")
        return block
    else:
        raise ValueError(f"Filtering mode {params['mode']} not recognized")


def detect_peaks(
    block_filtered: np.ndarray,
    channels_count: int,
    peak_height_threshold: float,
    min_peak_distance: int,
):
    """
    Run peak-finding for each channel in a filtered block.
    Returns a list of peak indices and a parallel list of channel indices.
    """
    log.debug("Starting peak detection")
    peaks_list = [
        find_peaks(
            block_filtered[:, ch],
            prominence=peak_height_threshold,
            distance=min_peak_distance,
        )[0]
        for ch in range(channels_count)
    ]
    channels_list = [
        np.array([ch] * len(peaks_list[ch]), dtype=np.int32)
        for ch in range(channels_count)
    ]
    return peaks_list, channels_list


def group_peaks(peaks, channels, peak_window):
    """
    Group peaks that lie within a certain distance (peak_window).
    Only merges them if they belong to different channels.
    """
    # Flatten out everything and sort
    peaks = np.concatenate(peaks)
    channels = np.concatenate(channels)
    sorter = np.argsort(peaks)
    peaks = peaks[sorter]
    channels = channels[sorter]

    peak_groups = []
    peak_channels = []
    if len(peaks) == 0:
        return peak_groups, peak_channels

    # Initialize the first group
    current_group = [peaks[0]]
    current_channels = [channels[0]]
    for p, c in zip(peaks[1:], channels[1:]):
        # Check if the current peak is within the peak_window of the last peak
        # in the current group, and if channel is new
        if ((p - current_group[-1]) < peak_window) and (c not in current_channels):
            current_group.append(p)
            current_channels.append(c)
        else:
            peak_groups.append(current_group)
            peak_channels.append(current_channels)
            current_group = [p]
            current_channels = [c]

    # Append the final group
    peak_groups.append(current_group)
    peak_channels.append(current_channels)

    # Check duplicates
    for i in range(len(peak_groups)):
        unique_channels, counts = np.unique(peak_channels[i], return_counts=True)
        if np.any(counts > 1):
            # Detected a peak group that contains the same channel multiple times
            log.warning(
                "Found duplicate channels in a peak group! This is a bug and should be reported."
            )

    return peak_groups, peak_channels


def filter_peak_groups(grouped_peaks, grouped_channels, min_channels_with_peaks):
    """
    Discard groups that do not have enough channels with peaks.
    """
    group_lengths = [len(g) for g in grouped_peaks]
    bool_idx = np.array(group_lengths) > min_channels_with_peaks
    grouped_channels = np.array(grouped_channels, dtype="object")[bool_idx]
    grouped_peaks = np.array(grouped_peaks, dtype="object")[bool_idx]
    return grouped_peaks, grouped_channels


def compute_mean_peak(
    block_filtered: np.ndarray,
    center: float,
    channels: np.ndarray,
    around_peak_window: int,
) -> Optional[np.ndarray]:
    """
    Given a filtered block, a center index, and the channels that had peaks,
    return a normalized mean-peak waveform across those channels.
    """
    indexer = np.arange(
        center - around_peak_window, center + around_peak_window, dtype=np.int32
    )
    if (np.any(indexer < 0)) or (np.any(indexer >= block_filtered.shape[0])):
        log.warning("Index out of bounds for peak window, skipping peak.")
        return None

    channels = np.array(channels, dtype=np.int32)

    try:
        peak_snippet = block_filtered[indexer][:, channels]
    except:
        print(indexer)
        print(channels)
        print(block_filtered.shape)
        embed()
        exit()

    # Pull each channel baseline to zero
    baseline_window = around_peak_window // 4
    baselines = np.mean(peak_snippet[:baseline_window], axis=0)
    peak_snippet -= baselines

    # Extract sign of strongest deviation
    signs = np.array([1 if s[np.argmax(np.abs(s))] > 0 else -1 for s in peak_snippet.T])

    # Flip sign according to majority polarity
    # TODO: This does not work sometimes, maybe due to noise? For single peak pulses
    # this works well but for multiphase peaks we should check the order of signs
    # of each single peak instead of the maximum because the maximum is subject to noise
    if np.all(signs > 0):
        log.debug("All peaks are positive")
        mean_peak = np.mean(peak_snippet, axis=1)
    elif np.all(signs < 0):
        log.debug("All peaks are negative")
        mean_peak = -np.mean(peak_snippet, axis=1)
    else:
        log.debug("Peaks are mixed, flipping signs")
        mean_peak = np.mean(peak_snippet * signs, axis=1)

    # Pull the combined baseline to zero
    start_baseline = np.mean(mean_peak[:baseline_window])
    mean_peak -= start_baseline

    # Normalize
    denominator = np.max(mean_peak) - np.min(mean_peak)
    if denominator == 0:
        log.warning("All values are the same, cannot normalize")
        return None
    mean_peak = mean_peak / denominator

    return mean_peak


def process_file(
    data: AudioLoader,
    save_path: Path,
    path: Path,
    buffersize_s: int = 60,
    min_channels_with_peaks: int = 4,
    smoothing_window_s: float = 0.0001,
    peak_distance_s: float = 0.004,
    peak_height_threshold: float = 0.001,
    resample: bool = True,
    n_resamples: int = 512,
):
    data.set_unwrap(thresh=1.5)
    blocksize = int(np.ceil(data.rate * buffersize_s))
    overlap = blocksize // 10  # Just a bit clearer than int(np.ceil(blocksize // 10))

    # Configuration
    min_peak_distance = int(np.ceil(peak_distance_s * data.rate))
    around_peak_window = int(np.round(0.75 * min_peak_distance))
    window_length = int(np.ceil(smoothing_window_s * data.rate))
    polyorder = 3
    filtering_params = dict(
        mode="savgol",
        window_length=window_length if window_length > polyorder else polyorder + 1,
        polyorder=3,
    )

    log.info(
        f"""
        Processing dataset: {data.filepath}.
        Recording duration: {pretty_duration_humanize(data.shape[0] / data.rate)}.
        With a sampling rate: {data.rate} Hz.
        Recorded on {data.channels} channels.
        Set buffersize_s: {buffersize_s} seconds, or {blocksize} samples.
        Set block_overlap to {overlap} samples.
        Set the time window of interest to {around_peak_window * 2 / data.rate} seconds.
        Set minimum peak distance: {peak_distance_s} seconds, or {min_peak_distance} samples.
        Set minimum peak height: {peak_height_threshold}.
        Set minimum channels with peaks: {min_channels_with_peaks}.
        Set smoothing window: {smoothing_window_s} seconds, or {filtering_params["window_length"]} samples.
        Set resampling to {resample} with {n_resamples} samples.
        """
    )

    dataset = initialize_dataset(path)
    collected_peak_counter = 0
    dataset_counter = 0

    num_blocks = int(np.ceil(data.shape[0] / (blocksize - overlap)))
    with get_progress() as pbar:
        desc = "Processing file"
        task = pbar.add_task(desc, total=num_blocks, transient=True)
        for blockiterval, block in enumerate(
            data.blocks(block_size=blocksize, noverlap=overlap)
        ):
            # Ensure we treat mono data as [n_samples, 1]
            if data.channels == 1:
                block = np.expand_dims(block, axis=1)

            # Apply filtering
            block_filtered = apply_filter(block, data.rate, params=filtering_params)

            # Detect peaks on each channel
            abs_block_filtered = np.abs(block_filtered)
            peaks_list, channels_list = detect_peaks(
                abs_block_filtered,
                data.channels,
                peak_height_threshold,
                min_peak_distance,
            )

            # Group them
            grouped_peaks, grouped_channels = group_peaks(
                peaks_list, channels_list, peak_window=int(min_peak_distance // 2)
            )

            # Filter out groups that do not meet the threshold
            grouped_peaks, grouped_channels = filter_peak_groups(
                grouped_peaks, grouped_channels, min_channels_with_peaks
            )

            # Compute means of each group
            log.info(f"Found a total of {len(grouped_peaks)} peaks")
            if len(grouped_peaks) == 0:
                log.debug(f"Found 0 peaks. Skipping block {blockiterval} for now.")
                continue

            centers = [int(np.mean(g)) for g in grouped_peaks]

            # For each peak group, compute & store waveform
            for peakiterval, (pks, chans, center) in enumerate(
                zip(grouped_peaks, grouped_channels, centers)
            ):
                chans = np.array(chans)
                if chans.dtype not in [np.int32, np.int64, np.bool]:
                    # This should not happen, but just in case
                    log.warning(
                        f"Channel type is not int32 or int64 but {chans.dtype}. Converting to int32."
                    )
                    chans = np.array(chans, dtype=np.int32)

                pks = np.array(pks)
                if pks.dtype not in [np.int32, np.int64]:
                    # This should not happen, but just in case
                    log.warning(
                        f"Peak type is not int32 or int64 but {pks.dtype}. Converting to int32."
                    )
                    pks = np.array(pks, dtype=np.int32)

                mean_peak = compute_mean_peak(
                    block_filtered, center, chans, around_peak_window
                )

                if (mean_peak is None) or (len(chans) == 0):
                    msg = "Failed to compute mean peak due to index out of range or degenerate peak shape."
                    log.warning(msg)
                    continue

                # Resample mean peak
                if resample:
                    log.debug("Resampling mean peak")
                    x = np.linspace(0, len(mean_peak), len(mean_peak))
                    f = interp1d(x, mean_peak, kind="cubic")
                    xnew = np.linspace(0, len(mean_peak), n_resamples)
                    mean_peak = f(xnew)

                # Mark which channels contributed
                bool_channels = np.zeros(data.channels, dtype=bool)
                bool_channels[chans] = True

                print("-------------------------------------")
                print("data.channels:")
                print(data.channels)
                print("pks:")
                print(pks)
                print("chans:")
                print(chans)
                print("block_filtered.shape:")
                print(block_filtered.shape)

                # Amplitudes on each channel
                amp_index = np.array(
                    [
                        np.argmax(np.abs(block_filtered[pks, ch]))
                        for ch in range(data.channels)
                    ]
                )
                amps = np.array(
                    [block_filtered[pks, ch][idx] for ch, idx in enumerate(amp_index)]
                )

                center += blockiterval * (blocksize - overlap)
                start_stop_index = [
                    center - around_peak_window,
                    center + around_peak_window,
                ]
                dataset["peaks"].append(mean_peak)
                dataset["channels"].append(bool_channels)
                dataset["amplitudes"].append(amps)
                dataset["centers"].append(center)
                dataset["start_stop_index"].append(start_stop_index)
                collected_peak_counter += 1

            pbar.update(task, advance=1)

        dataset["peaks"] = np.array(dataset["peaks"])
        dataset["channels"] = np.array(dataset["channels"])
        dataset["amplitudes"] = np.array(dataset["amplitudes"])
        dataset["centers"] = np.array(dataset["centers"])
        dataset["start_stop_index"] = np.array(dataset["start_stop_index"])
        dataset["rate"] = data.rate

        save_numpy(dataset, save_path)


@app.command()
def main(
    datapath: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
    filetype: Annotated[
        str, typer.Option("--filetype", "-f", help="File type to detect peaks on.")
    ] = "wav",
    buffersize_seconds: Annotated[
        int, typer.Option("--buffersize", "-b", help="Buffer size in seconds.")
    ] = 60,
    min_channels_with_peaks: Annotated[
        int, typer.Option("--min_channels", "-mc", help="Minimum channels with peaks.")
    ] = 4,
    smoothing_window_s: Annotated[
        float,
        typer.Option("--smoothing_window", "-sw", help="Smoothing window in seconds."),
    ] = 0.0001,
    peak_distance_s: Annotated[
        float,
        typer.Option(
            "--peak_distance", "-pd", help="Minimum distance between peaks in seconds."
        ),
    ] = 0.004,
    peak_height_threshold: Annotated[
        float,
        typer.Option(
            "--peak_height", "-ph", help="Minimum height of peaks in amplitude."
        ),
    ] = 0.001,
    resample: Annotated[
        bool, typer.Option("--resample", "-r", help="Resample the data.")
    ] = True,
    n_resamples: Annotated[
        int, typer.Option("--n_resamples", "-nr", help="Number of resamples.")
    ] = 512,
):
    """
    Main function to orchestrate:
    1) Data loading
    2) Filtering
    3) Peak detection
    4) Grouping
    5) Waveform extraction
    6) Dataset saving
    """

    savepath = (
        None  # Maybe for later when we want to save the extracted peaks somewhere else
    )
    configure_logging(verbosity=verbose)
    file_list, save_list = load_raw_data(path=datapath, filetype=filetype)
    for data, save_path in zip(file_list, save_list):
        data = AudioLoader(data)
        process_file(
            data,
            save_path,
            datapath,
            buffersize_seconds,
            min_channels_with_peaks,
            smoothing_window_s,
            peak_distance_s,
            peak_height_threshold,
            resample,
            n_resamples,
        )
