"""Main loop to parse large datasets and collect peaks."""

import gc
from datetime import timedelta
from pathlib import Path
from typing import Annotated
import json

import humanize
import numpy as np
import typer
from audioio.audioloader import AudioLoader
from IPython import embed
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from thunderpulse.data_handling.data import load_data, get_file_list, Data
from thunderpulse.pulse_detection.config import (
    FiltersParameters,
    FindPeaksKwargs,
    Params,
    filter_map,
    pretty_print_config,
)
from thunderpulse.utils.loggers import (
    configure_logging,
    get_logger,
    get_progress,
)

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)


def pretty_duration_humanize(seconds: float) -> str:
    """Return a human-friendly string of a duration given in seconds."""
    return humanize.naturaldelta(timedelta(seconds=seconds))


def initialize_dataset() -> dict:
    """Prepare structure into which extracted peaks/metadata will be placed."""
    log.debug("Creating empty dataset to save extracted peaks")
    return {
        "peaks": [],
        "channels": [],
        "amplitudes": [],
        "centers": [],
        "start_stop_index": [],
        "rate": None,
    }


def apply_filters(
    data: np.ndarray, rate: float, params: FiltersParameters
) -> np.ndarray:
    """Apply the specified filters to the data."""
    if not isinstance(params.filters, list):
        msg = "Filters must be a list of strings"
        raise TypeError(msg)
    if not isinstance(params.filter_params, list):
        msg = "Filter parameters must be a list of dictionaries"
        raise TypeError(msg)
    if len(params.filters) != len(params.filter_params):
        msg = "Filters and filter parameters must be the same length"
        raise ValueError(msg)
    if len(params.filters) == 0:
        log.debug("No filters applied")
        return data

    # Apply all filters in sequence
    for filter_name, filter_params in zip(
        params.filters,
        params.filter_params,
        strict=True,
    ):
        log.debug(f"Applying filter: {filter_name}")
        data = filter_map[filter_name](
            data,
            **filter_params.to_kwargs(keep_none=True),
        )
    return data


def detect_peaks(
    block: np.ndarray,
    mode: str = "both",
    find_peaks_kwargs: FindPeaksKwargs | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect peaks in 1D signal.

    Parameters
    ----------
    block_filtered : np.ndarray
        Array with shape ``(n_samples, channels_count)`` containing the
        pre-filtered signal.
    mode : {'peak', 'trough', 'both'}, default 'both'
        * ``'peak'``   – detect positive excursions only
        * ``'trough'`` – detect negative excursions only
        * ``'both'``   – detect both and merge the results (default)
    find_peaks_kwargs : dict, optional
        Additional arguments passed to `scipy.signal.find_peaks`.

    Returns
    -------
    peaks_list : list[np.ndarray]
        One NumPy array per channel containing peak sample-indices (sorted).
    channels_list : list[np.ndarray]
        Parallel list containing the channel-ID for every detected index.
    """
    if mode not in {"peak", "trough", "both"}:
        msg = "Mode must be one of {'peak', 'trough', 'both'}"
        raise ValueError(msg)

    log.debug(f"Starting peak detection with mode: {mode}")

    peaks_list = []
    channels_list = []

    for ch in range(block.shape[1]):
        signal = block[:, ch]

        if find_peaks_kwargs:
            peak_params = find_peaks_kwargs.to_kwargs(keep_none=True)
        else:
            peak_params = {}

        neg_peaks = np.empty(0, dtype=int)
        pos_peaks = np.empty(0, dtype=int)

        if mode in {"peak", "both"}:
            pos_peaks, _ = find_peaks(signal, **peak_params)
        if mode in {"trough", "both"}:
            neg_peaks, _ = find_peaks(-signal, **peak_params)

        # concatenate & sort to preserve chronological order
        peaks_ch = np.sort(np.concatenate((pos_peaks, neg_peaks)))
        peaks_list.append(peaks_ch)

        # build matching channel-id array
        channels_list.append(np.full_like(peaks_ch, ch, dtype=np.int32))

    return peaks_list, channels_list


def group_peaks_across_channels_by_time(
    peaks: list[np.ndarray], channels: list[np.ndarray], peak_window: int = 10
) -> tuple:
    """Group peaks that lie within a certain distance (peak_window).

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
    for p, c in zip(peaks[1:], channels[1:], strict=False):
        # Check if the current peak is within the peak_window of the last peak
        # in the current group, and if channel is new
        if ((p - current_group[-1]) < peak_window) and (
            c not in current_channels
        ):
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
        _, counts = np.unique(peak_channels[i], return_counts=True)
        if np.any(counts > 1):
            msg = (
                "Found duplicate channels in a peak group! "
                f"Peak group: {peak_groups[i]}, Channels: {peak_channels[i]}"
            )
            log.warning(msg)

    return peak_groups, peak_channels


def filter_peak_groups(
    grouped_peaks: list[np.ndarray],
    grouped_channels: list[np.ndarray],
    min_channels_with_peaks: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Filter out peak groups that are found on too few channels.

    Parameters
    ----------
    grouped_peaks
        List of 1-D index arrays – one array per detected peak group.
    grouped_channels
        List of 1-D channel-index arrays matching `grouped_peaks`.
    min_channels_with_peaks
        A group is **kept** only if it is present on *strictly more than* this
        number of channels.

    Returns
    -------
    kept_peaks, kept_channels
        Two lists (same length, same order) containing only the qualifying
        groups.
    """
    kept_peaks: list[np.ndarray] = []
    kept_channels: list[np.ndarray] = []

    for peaks, chans in zip(grouped_peaks, grouped_channels, strict=True):
        if len(peaks) > min_channels_with_peaks:
            kept_peaks.append(peaks)
            kept_channels.append(chans)

    return kept_peaks, kept_channels


def compute_mean_peak(
    block_filtered: np.ndarray,
    center: float,
    channels: np.ndarray,
    around_peak_window: int,
) -> np.ndarray | None:
    """
    Given a filtered block, a center index, and the channels that had peaks,
    return a normalized mean-peak waveform across those channels.
    """
    indexer = np.arange(
        center - around_peak_window,
        center + around_peak_window,
        dtype=np.int32,
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
    signs = np.array(
        [1 if s[np.argmax(np.abs(s))] > 0 else -1 for s in peak_snippet.T]
    )

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


def process_block(
    input_data: np.ndarray, rate: float, params: Params, blockinfo: dict
):
    output_data = {
        "peaks": [],
        "channels": [],
        "amplitudes": [],
        "centers": [],
        "start_stop_index": [],
    }

    # Ensure we treat mono data as [n_samples, 1]
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=1)

    n_channels = input_data.shape[1]

    # Apply filtering
    block_filtered = apply_filters(input_data, rate, params=params.filters)

    # Detect peaks on each channel
    peaks_list, channels_list = detect_peaks(
        block_filtered,
        mode=params.peaks.mode,
        find_peaks_kwargs=params.peaks.find_peaks_kwargs,
    )

    # Convert config paramters from seconds to samples
    min_peak_distance = int(np.ceil(params.peaks.min_peak_distance_s * rate))
    cutout_window_around_peak = int(
        np.ceil(params.peaks.cutout_window_around_peak_s * rate)
    )

    # TODO: Electrode distance sorting from Alex

    # Group peaks across channels when they are close in time
    grouped_peaks, grouped_channels = group_peaks_across_channels_by_time(
        peaks_list,
        channels_list,
        peak_window=min_peak_distance,
    )

    # Filter out groups that do not meet the threshold
    if params.peaks.min_channels > 1:
        grouped_peaks, grouped_channels = filter_peak_groups(
            grouped_peaks, grouped_channels, params.peaks.min_channels
        )

    # Compute means of each group
    log.info(f"Found a total of {len(grouped_peaks)} peaks")

    if len(grouped_peaks) == 0:
        log.debug(
            f"Found 0 peaks. Skipping block {blockinfo['blockiterval']} for now."
        )
        return None

    centers = [int(np.mean(g)) for g in grouped_peaks]

    # For each peak group, compute & store waveform
    peak_counter = 0
    for _, (pks, chans, center) in enumerate(
        zip(grouped_peaks, grouped_channels, centers, strict=False)
    ):
        chans = np.array(chans)
        pks = np.array(pks)

        if chans.dtype not in [np.int32, np.int64, np.bool]:
            msg = f"Channel variable type ('chans') is not int32 or int64 but {chans.dtype}"
            raise TypeError(msg)

        if pks.dtype not in [np.int32, np.int64]:
            msg = "Peak variable type ('pks') is not int32 or int64 but {pks.dtype}"
            raise TypeError(msg)

        mean_peak = compute_mean_peak(
            block_filtered,
            center,
            chans,
            cutout_window_around_peak,
        )

        if (mean_peak is None) or (len(chans) == 0):
            msg = "Failed to compute mean peak due to index out of range or degenerate peak shape."
            log.warning(msg)
            continue

        # Resample mean peak
        if params.resample.enabled:
            log.debug("Resampling mean peak")
            x = np.linspace(0, len(mean_peak), len(mean_peak))
            f = interp1d(x, mean_peak, kind="cubic")
            xnew = np.linspace(0, len(mean_peak), params.resample.n_resamples)
            mean_peak = f(xnew)

        # Mark which channels contributed
        bool_channels = np.zeros(n_channels, dtype=bool)
        bool_channels[chans] = True

        # Amplitudes on each channel
        amp_index = np.array(
            [
                np.argmax(np.abs(block_filtered[pks, ch]))
                for ch in range(n_channels)
            ]
        )
        amps = np.array(
            [block_filtered[pks, ch][idx] for ch, idx in enumerate(amp_index)]
        )

        center = center + blockinfo["blockiterval"] * (
            blockinfo["blocksize"] - blockinfo["overlap"]
        )
        start_stop_index = [
            center - cutout_window_around_peak,
            center + cutout_window_around_peak,
        ]

        peak_counter += 1
        output_data["peaks"].append(mean_peak)
        output_data["channels"].append(bool_channels)
        output_data["amplitudes"].append(amps)
        output_data["centers"].append(center)
        output_data["start_stop_index"].append(start_stop_index)

    return output_data, peak_counter


def process_file(
    data: Data,
    # save_path: Path,
    params: Params,
) -> None:
    if data.metadata.samplerate is None:
        msg = "Data rate is None, cannot process file."
        raise ValueError(msg)
    rate = data.metadata.samplerate

    blocksize = int(np.ceil(rate * params.buffersize_s))
    overlap = blocksize // 10

    num_blocks = int(np.ceil(data.metadata.frames / (blocksize - overlap)))
    with get_progress() as pbar:
        desc = "Processing file"
        task = pbar.add_task(desc, total=num_blocks, transient=True)
        for blockiterval, block in enumerate(
            data.blocks(blocksize=blocksize, overlap=overlap)
        ):
            blockinfo = {
                "blockiterval": blockiterval,
                "blocksize": blocksize,
                "overlap": overlap,
            }

            # TODO: Why is my linter crying about this?
            block_peaks, peak_counter = process_block(
                block, rate, params, blockinfo
            )

            log.info(
                f"Processed block {blockiterval} with {peak_counter} peaks detected."
            )

            pbar.update(task, advance=1)

        # TODO: Here Nix file saving instead of Numpy
        gc.collect()  # TODO: Check if this has actually an effect


@app.command()
def main(
    datapath: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    probepath: Annotated[Path, typer.Option(help="Path to the probe file.")],
    configpath: Annotated[Path, typer.Option(help="Path to the config file.")],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", "-o", help="Overwrite existing files."),
    ] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
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

    configure_logging(verbosity=verbose)

    params = Params().from_json(str(configpath))
    subdirs = sorted(datapath.glob("*/"))

    for recording_path in subdirs:
        # TODO: Re-implement this for *nix files
        # Skip if file already exists and overwrite is not set
        # if save_path.with_suffix(".npz").exists() and not overwrite:
        #     log.info(f"File {save_path} already exists, skipping.")
        #     continue

        try:
            data = load_data(recording_path, probepath)
        except Exception as e:
            log.error(f"Failed to load {recording_path}: {e}")
            continue

        process_file(
            data,
            # save_path,
            params=params,
        )


if __name__ == "__main__":
    app()
