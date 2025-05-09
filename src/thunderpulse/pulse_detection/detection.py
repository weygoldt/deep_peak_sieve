"""Main loop to parse large datasets and collect pulses."""

import gc
import sys
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Iterable
from IPython import embed

import humanize
import matplotlib.pyplot as plt
import nixio
import numpy as np
import typer
from IPython import embed
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from thunderpulse.data_handling.data import Data, SensorArray, load_data
from thunderpulse.dsp.common_reference import common_median_reference
from thunderpulse.pulse_detection.config import (
    FiltersParameters,
    FindPeaksKwargs,
    Params,
    PrefilterParameters,
    filter_map,
)
from thunderpulse.utils.loggers import (
    configure_logging,
    get_logger,
)

log = get_logger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


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
    data: np.ndarray,
    rate: float,
    params: Params,
) -> np.ndarray:
    """Apply the specified filters to the data."""
    if not isinstance(params.filters.filters, list):
        msg = "Filters must be a list of strings"
        raise TypeError(msg)
    if not isinstance(params.filters.filter_params, list):
        msg = "Filter parameters must be a list of dictionaries"
        raise TypeError(msg)
    if len(params.filters.filters) != len(params.filters.filter_params):
        msg = "Filters and filter parameters must be the same length"
        raise ValueError(msg)
    if len(params.filters.filters) == 0:
        log.debug("No filters applied")
        return data

    # PreFilter operations
    # TODO: This should be a list of operations to apply before filtering
    if params.prefitering.common_median_reference:
        data = common_median_reference(data)

    # Apply all filters in sequence
    log.info("Applying filters")
    for filter_name, filter_params in zip(
        params.filters.filters,
        params.filters.filter_params,
        strict=True,
    ):
        log.debug(f"Applying filter: {filter_name}")
        log.debug(f"Filter parameters: {filter_params}")

        # Check if filter requires samplerate
        data = filter_map[filter_name](
            data,
            rate,
            **filter_params.to_kwargs(keep_none=False),
        )
        log.debug(f"Filter {filter_name} applied")

    return data


def detect_peaks(
    block: np.ndarray,
    rate: float,
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

    log.info(f"Starting peak detection with mode: {mode}")

    peaks_list = []
    channels_list = []

    if find_peaks_kwargs:
        peak_params = find_peaks_kwargs.to_kwargs(keep_none=True)
    else:
        peak_params = {}

    # Convert parameters from seconds to samples
    for key, value in peak_params.items():
        # Select only the parameters that are x-axis related
        if not key in ["width", "distance"]:
            continue
        if isinstance(value, Iterable) and not isinstance(value, str):
            peak_params[key] = [x / rate for x in value]
        elif isinstance(value, (int | float)):
            peak_params[key] = value * rate
        elif value is None:
            pass
        else:
            msg = (
                "Peak parameters must be either a list of numbers or a "
                f"single number. Got {type(value)} for key {key}."
            )
            raise TypeError(msg)

        print(f"Peak parameter {key}: {peak_params[key]}")

    # Detect peaks on each channel
    for ch in range(block.shape[1]):
        signal = block[:, ch]

        peaks = np.zeros(0, dtype=np.int32)
        if mode == "peak":
            peaks, _ = find_peaks(signal, **peak_params)
        if mode == "trough":
            peaks, _ = find_peaks(-signal, **peak_params)
        if mode == "both":
            peaks, _ = find_peaks(np.abs(signal), **peak_params)

        # concatenate & sort to preserve chronological order
        peaks = np.sort(peaks)
        peaks_list.append(peaks)

        # build matching channel-id array
        channels_list.append(np.full_like(peaks, ch, dtype=np.int32))

    return peaks_list, channels_list


def group_peaks_across_channels_by_time(
    peaks: list[np.ndarray], channels: list[np.ndarray], peak_window: int = 10
) -> tuple:
    """Group peaks that lie within a certain distance (peak_window).

    Only merges them if they belong to different channels.
    """
    log.info("Grouping peaks across channels")
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


def filter_peak_groups_by_distance(
    grouped_peaks: list[np.ndarray],
    grouped_channels: list[np.ndarray],
    sensoryarray: SensorArray,
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
    log.info("Filtering peak groups by distance")
    kept_peaks: list[np.ndarray] = []
    kept_channels: list[np.ndarray] = []
    close_channels = sensoryarray.ids[np.argsort(sensoryarray.y)]

    points = np.vstack((sensoryarray.x, sensoryarray.y)).T
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)

    for peaks, chans in zip(grouped_peaks, grouped_channels, strict=True):
        # if len(peaks) >= min_channels_with_peaks:
        close_channels = np.argsort(distances[chans])[
            :, 1 : min_channels_with_peaks + 1
        ]
        new_grouping_bool = (
            close_channels[:, np.newaxis] == np.array(chans)[:, np.newaxis]
        )
        new_grouping_index = np.where(new_grouping_bool)
        _, first_dim_count = np.unique(
            new_grouping_index[0], return_counts=True
        )
        if new_grouping_index[0].size == 0:
            peaks = [[p] for p in peaks]
            chans = [[ch] for ch in chans]
            kept_peaks.extend(np.array(peaks))
            kept_channels.extend(np.array(chans))
            continue

        new_group_chan = np.split(
            np.array(chans)[new_grouping_index[1]],
            np.cumsum(first_dim_count[:-1]),
        )
        new_group_peaks = np.split(
            np.array(peaks)[new_grouping_index[1]],
            np.cumsum(first_dim_count[:-1]),
        )

        new_peaks = [new_group_peaks[0]]
        new_chans = [new_group_chan[0]]
        for ngc, ngp in zip(
            new_group_chan[1:], new_group_peaks[1:], strict=True
        ):
            check_if_in_group = ~np.isin(
                ngc, np.concatenate(new_chans).flatten()
            )
            if np.all(check_if_in_group):
                new_chans.append(ngc)
                new_peaks.append(ngp)
                continue
            if np.any(check_if_in_group):
                new_chans[-1] = np.append(
                    new_chans[-1], ngc[check_if_in_group]
                )
                new_peaks[-1] = np.append(
                    new_peaks[-1], ngp[check_if_in_group]
                )
                continue

        min_channels_index = np.where(
            np.array([len(c) for c in new_peaks]) >= min_channels_with_peaks
        )[0]

        if min_channels_index.size > 0:
            for index in min_channels_index:
                kept_peaks.append(new_peaks[index])
                kept_channels.append(new_chans[index])

    return kept_peaks, kept_channels


def filter_peak_groups_by_count(
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
    log.info("Filtering peak groups")
    kept_peaks: list[np.ndarray] = []
    kept_channels: list[np.ndarray] = []

    for peaks, chans in zip(grouped_peaks, grouped_channels, strict=True):
        if len(peaks) >= min_channels_with_peaks:
            kept_peaks.append(peaks)
            kept_channels.append(chans)

    return kept_peaks, kept_channels


def compute_mean_peak(
    peak: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    """Given a filtered block, a center index, and the channels that had peaks,
    return a normalized mean-peak waveform across those channels.
    """

    peak = peak[channels, :]

    # Pull each channel baseline to zero
    baseline_window = peak.shape[-1] // 4
    baselines = np.mean(peak[:, :baseline_window], axis=-1)
    peak -= baselines.reshape(-1, 1)

    # Extract sign of strongest deviation
    signs = np.array([1 if s[np.argmax(np.abs(s))] > 0 else -1 for s in peak])

    # Flip sign according to majority polarity
    # TODO: This does not work sometimes, maybe due to noise? For single peak pulses
    # this works well but for multiphase peaks we should check the order of signs
    # of each single peak instead of the maximum because the maximum is subject to noise

    if np.all(signs > 0):
        log.debug("All peaks are positive")
        mean_peak = np.mean(peak, axis=0)
    elif np.all(signs < 0):
        log.debug("All peaks are negative")
        mean_peak = -np.mean(peak, axis=0)
    else:
        log.debug("Peaks are mixed, flipping signs")
        mean_peak = np.mean(peak * signs.reshape(-1, 1), axis=0)

    # Pull the combined baseline to zero again
    start_baseline = np.mean(mean_peak[:baseline_window])
    mean_peak -= start_baseline

    return mean_peak


def detect_peaks_on_block(
    input_data: np.ndarray,
    rate: float,
    blockinfo: dict,
    params: Params,
) -> dict | None:
    n_channels = input_data.shape[1]
    output_data = {}

    # Apply filtering
    block_filtered = apply_filters(input_data, rate, params=params)

    # Detect peaks on each channel
    peaks_list, channels_list = detect_peaks(
        block_filtered,
        rate=rate,
        mode=params.peaks.mode,
        find_peaks_kwargs=params.peaks.find_peaks_kwargs,
    )

    # NOTE: Take default Values if is None
    # Convert config paramters from seconds to samples
    if not params.peaks.min_peak_distance_s:
        min_peak_distance = int(
            np.ceil(Params().peaks.min_peak_distance_s * rate)
        )
    else:
        min_peak_distance = int(
            np.ceil(params.peaks.min_peak_distance_s * rate)
        )

    if not params.peaks.cutout_window_around_peak_s:
        cutout_window_around_peak = int(
            np.ceil(Params().peaks.cutout_window_around_peak_s * rate)
        )
    else:
        cutout_window_around_peak = int(
            np.ceil(params.peaks.cutout_window_around_peak_s * rate)
        )

    # Group peaks across channels when they are close in time
    grouped_peaks, grouped_channels = group_peaks_across_channels_by_time(
        peaks_list,
        channels_list,
        peak_window=min_peak_distance,
    )

    # output_data["all_pulses"] = grouped_peaks
    # output_data["all_channels"] = grouped_channels

    # Filter out groups that do not meet the threshold
    if not params.peaks.min_channels:
        min_channels = Params().peaks.min_channels
    else:
        min_channels = params.peaks.min_channels

    grouped_peaks, grouped_channels = filter_peak_groups_by_distance(
        grouped_peaks,
        grouped_channels,
        params.sensoryarray,
    )

    if min_channels > 1:
        grouped_peaks, grouped_channels = filter_peak_groups_by_count(
            grouped_peaks,
            grouped_channels,
            min_channels_with_peaks=min_channels,
        )

    # Compute means of each group
    log.info(f"Found a total of {len(grouped_peaks)} peaks")

    if len(grouped_peaks) == 0:
        msg = "No peaks found in block, skipping."
        log.debug(msg)
        log.warning("No peaks detected")
        return None

    # output_data["all_pulses_groups"] = grouped_peaks
    # output_data["all_channels_groups"] = grouped_channels

    centers = [int(np.mean(g)) for g in grouped_peaks]

    # For each peak group, compute & store waveform
    peak_counter = 0
    cutout_size = cutout_window_around_peak * 2
    peak_array = np.full(
        shape=(len(grouped_peaks), n_channels, cutout_size),
        fill_value=np.nan,
        dtype=np.float32,
    )
    channels_array = np.full(
        shape=(len(grouped_peaks), n_channels), fill_value=False, dtype=bool
    )
    centers_array = np.full(
        shape=(len(grouped_peaks),), fill_value=-1, dtype=np.int32
    )
    start_stop_index = np.full(
        shape=(len(grouped_peaks), 2), fill_value=-1, dtype=np.int32
    )

    output_data["pulses"] = peak_array
    output_data["channels"] = channels_array
    output_data["centers"] = centers_array
    output_data["start_stop_index"] = start_stop_index

    log.info("Cutting out peaks")

    for _, (pks, chans, center) in enumerate(
        zip(grouped_peaks, grouped_channels, centers, strict=False)
    ):
        # Mark which channels contributed
        chans = np.array(chans)
        bool_channels = np.zeros(n_channels, dtype=bool)
        bool_channels[chans] = True

        start_stop_index = [
            center - cutout_window_around_peak,
            center + cutout_window_around_peak,
        ]

        if (start_stop_index[0] < 0) or (
            start_stop_index[1] > input_data.shape[0]
        ):
            log.warning("Peak is too close to recording borders, skipping.")
            continue

        p = input_data[start_stop_index[0] : start_stop_index[1], :].T
        if p.shape[1] != cutout_window_around_peak * 2:
            log.critical("This is not supposed to happen")
            continue

        peak_counter += 1

        # Correct to global time, not just block time
        center = center + blockinfo["blockiterval"] * (
            blockinfo["blocksize"] - blockinfo["overlap"]
        )

        # Same for start/stop indices
        start_stop_index = [
            center - cutout_window_around_peak,
            center + cutout_window_around_peak,
        ]

        output_data["pulses"][peak_counter - 1, :] = p
        output_data["channels"][peak_counter - 1, :] = bool_channels
        output_data["centers"][peak_counter - 1] = center
        output_data["start_stop_index"][peak_counter - 1] = start_stop_index

    return output_data


def post_process_peaks_per_block(
    peaks: dict, params: Params, blockinfo: dict
) -> dict:
    n_peaks = peaks["pulses"].shape[0]
    n_channels = peaks["pulses"].shape[1]
    n_samples = peaks["pulses"].shape[2]

    # Interpolate raw peak snippets
    if params.resample.enabled:
        log.debug("Resampling peak snippets")
        new_shape = (n_peaks, n_channels, params.resample.n_resamples)
        new_peaks = np.full(
            shape=new_shape,
            fill_value=np.nan,
            dtype=peaks["pulses"].dtype,
        )

        for i in range(n_peaks):
            for ch in range(n_channels):
                x = np.linspace(0, n_samples, n_samples)
                xnew = np.linspace(0, n_samples, params.resample.n_resamples)
                f = interp1d(x, peaks["pulses"][i, ch], kind="cubic")
                new_peaks[i, ch] = f(xnew)

        peaks["pulses"] = new_peaks
        n_samples = params.resample.n_resamples

    # Compute mean peaks
    peaks["mean_pulses"] = np.full(
        shape=(n_peaks, n_samples),
        fill_value=np.nan,
        dtype=peaks["pulses"].dtype,
    )
    log.debug("Computing mean pulses")
    for i in range(len(peaks["pulses"])):
        mean_peak = compute_mean_peak(
            peaks["pulses"][i],
            peaks["channels"][i],
        )
        peaks["mean_pulses"][i] = mean_peak

    # TODO: Add option to normalize mean peaks between 0 and 1 or not
    # And add option to center the peak in the window using different methods
    # (max), max derivative, etc.

    # Normalize mean peaks
    if params.normalize.enabled:
        log.debug("Normalizing mean pulses")
        for i in range(len(peaks["mean_pulses"])):
            peaks["mean_pulses"][i] = peaks["mean_pulses"][i] / np.max(
                np.abs(peaks["mean_pulses"][i])
            )

    return peaks


def process_dataset(
    data: Data,
    params: Params,
    output_path: Path,
) -> None:
    if data.metadata.samplerate is None:
        msg = "Data rate is None, cannot process file."
        raise ValueError(msg)
    rate = data.metadata.samplerate
    blocksize = int(np.ceil(rate * params.buffersize_s))
    overlap = blocksize // 10

    # Open NIX file
    nix_file = nixio.File.open(str(output_path), nixio.FileMode.Overwrite)
    nix_block = nix_file.create_block(
        name="pulses.nix", type_="ThunderPulse.pulses"
    )

    num_blocks = int(np.ceil(data.metadata.frames / (blocksize - overlap)))
    # with get_progress() as pbar:
    #     desc = "Processing file"
    #     task = pbar.add_task(desc, total=num_blocks, transient=True)
    for blockiterval, block in enumerate(
        data.blocks(blocksize=blocksize, overlap=overlap)
    ):
        blockinfo = {
            "blockiterval": blockiterval,
            "blocksize": blocksize,
            "overlap": overlap,
        }

        block_peaks = detect_peaks_on_block(
            block,
            rate,
            blockinfo,
            params,
        )

        if block_peaks is None:
            log.info("Skipping block due to no peaks found")
            continue

        block_peaks = post_process_peaks_per_block(
            block_peaks, params, blockinfo
        )
        if blockiterval == 0:
            # Initialize dataset
            for key, value in block_peaks.items():
                try:
                    data_array = nix_block.create_data_array(
                        key, f"thunderpulse.{key}", data=value
                    )
                    print(f"Created Data array {data_array.name}")
                except ValueError:
                    embed()
                    exit()

        else:
            for key, value in block_peaks.items():
                try:
                    data_array = nix_block.data_arrays[key]
                    data_array.append(value)
                except IndexError:
                    data_array = nix_block.create_data_array(
                        key, f"thunderpulse.{key}", data=value
                    )
                    print(f"Created Data array {data_array.name}")

        # pbar.update(task, advance=1)

        gc.collect()  # TODO: Check if this has actually an effect
    nix_file.close()


@app.command()
def main(
    datapath: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    savepath: Annotated[
        Path | None, typer.Argument(help="Path to save the dataset.")
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", "-o", help="Overwrite existing files."),
    ] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 3,
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
    log.info("Starting ThunderPulse Pulse Detection")

    if savepath is None:
        savepath = datapath.parent / f"{str(datapath.name)}_pulses"
    savepath.mkdir(parents=True, exist_ok=True)

    possible_sensor_layout_names = ["electrode_layout.json", "probe.prb"]
    config_path = datapath / "config.json"
    # TODO: add probe path

    if not config_path.exists():
        msg = f"No config file found at {config_path}. Please provide a config file."
        raise FileNotFoundError(msg)

    sensor_layout_found = False
    sensor_layout_path = None
    log.info("Searching for sensor layout file")
    for sensor_layout_name in possible_sensor_layout_names:
        sensor_layout_path = datapath / sensor_layout_name
        if sensor_layout_path.exists():
            sensor_layout_found = True
            msg = f"Found sensor layout file: {sensor_layout_path}"
            log.info(msg)
            break
    if not sensor_layout_found or sensor_layout_path is None:
        msg = (
            "No sensor layout file found. Please provide a sensor layout file "
            "in the dataset directory."
            f" Possible names: {possible_sensor_layout_names}"
            f" Found: {sensor_layout_path}"
        )
        raise FileNotFoundError(msg)
    log.info(f"Using sensor layout file: {sensor_layout_path}")

    params = Params().from_json(str(config_path))
    subdirs = sorted(datapath.glob("*/"))

    for recording_path in subdirs:
        outfile_dir = savepath / recording_path.name
        outfile_dir.mkdir(parents=True, exist_ok=True)
        outfile_path = outfile_dir / "pulses.h5"
        if outfile_path.exists() and not overwrite:
            log.info(f"File {outfile_path} already exists, skipping.")
            continue
        try:
            data = load_data(recording_path, savepath, sensor_layout_path)
        except Exception as e:
            log.error(f"Failed to load {recording_path}: {e}")
            continue

        log.info(f"Processing file: {recording_path}")
        process_dataset(
            data,
            params=params,
            output_path=outfile_path,
        )
