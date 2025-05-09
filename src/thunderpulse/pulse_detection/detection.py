"""Main loop to parse large datasets and collect pulses."""

import gc
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Iterable

import humanize
import matplotlib.pyplot as plt
import nixio
import numpy as np
import typer
from IPython import embed
from nixio.util import check_entity_type
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from thunderpulse.data_handling.data import Data, SensorArray, load_data
from thunderpulse.dsp.common_reference import common_median_reference
from thunderpulse.pulse_detection.config import (
    FiltersParameters,
    FindPeaksKwargs,
    Params,
    PreProcessingParameters,
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
    if params.preprocessing.common_median_reference:
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
) -> tuple[np.ndarray, np.ndarray]:
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

    return np.concatenate(peaks_list), np.concatenate(channels_list)


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
    pulse_list, channels_list = detect_peaks(
        block_filtered,
        rate=rate,
        mode=params.peaks.mode,
        find_peaks_kwargs=params.peaks.find_peaks_kwargs,
    )

    if not params.peaks.cutout_window_around_peak_s:
        cutout_window_around_peak = int(
            np.ceil(Params().peaks.cutout_window_around_peak_s * rate)
        )
    else:
        cutout_window_around_peak = int(
            np.ceil(params.peaks.cutout_window_around_peak_s * rate)
        )

    cutout_size = cutout_window_around_peak * 2

    bad_pulse = np.where(
        (pulse_list - cutout_window_around_peak < 0)
        | (pulse_list + cutout_window_around_peak > input_data.shape[0])
    )[0]
    good_pulse_bool = np.ones_like(pulse_list, dtype=bool)
    good_pulse_bool[bad_pulse] = False

    start_stop_index = np.full(
        shape=(len(pulse_list), 2), fill_value=-1, dtype=np.int32
    )
    pulse_center = np.zeros_like(pulse_list)

    pulse_array = np.zeros((pulse_list.shape[0], cutout_size))

    for i, (peak, ch) in enumerate(
        zip(
            pulse_list[good_pulse_bool],
            channels_list[good_pulse_bool],
            strict=True,
        )
    ):
        pulse_index = np.arange(
            peak - cutout_window_around_peak,
            peak + cutout_window_around_peak,
        )

        pulse_array[i] = block_filtered[pulse_index, ch]

        pulse_center[i] = peak + blockinfo["blockiterval"] * (
            blockinfo["blocksize"] - blockinfo["overlap"]
        )

        # Same for start/stop indices
        start_stop_index[i, :] = [
            -cutout_window_around_peak + pulse_center[i],
            cutout_window_around_peak + pulse_center[i],
        ]

    output_data["pulses"] = pulse_array
    output_data["channels"] = channels_list
    output_data["centers"] = pulse_center
    output_data["start_stop_index"] = start_stop_index

    # TODO: Vectorize this
    # pulse_array = input_data[:, channels_list][
    #     peaks_list[:, np.newaxis]
    # ] + np.arange(-cutout_window_around_peak, cutout_window_around_peak)

    # NOTE: Take default Values if is None
    # Convert config paramters from seconds to samples
    # if not params.peaks.min_peak_distance_s:
    #     min_peak_distance = int(
    #         np.ceil(Params().peaks.min_peak_distance_s * rate)
    #     )
    # else:
    #     min_peak_distance = int(
    #         np.ceil(params.peaks.min_peak_distance_s * rate)
    #     )

    return output_data


def post_process_peaks_per_block(
    peaks: dict,
    params: Params,
    blockinfo: dict,
) -> dict:
    n_peaks = peaks["pulses"].shape[0]
    n_samples = peaks["pulses"].shape[1]

    # Interpolate raw peak snippets
    # TODO: DATATYOE TO INT
    if params.postprocessing.enable_resampling:
        log.debug("Resampling peak snippets")
        new_shape = (n_peaks, params.postprocessing.n_resamples)
        new_peaks = np.full(
            shape=new_shape,
            fill_value=np.nan,
            dtype=peaks["pulses"].dtype,
        )

        for i in range(n_peaks):
            x = np.linspace(0, n_samples, n_samples)
            xnew = np.linspace(0, n_samples, params.postprocessing.n_resamples)
            f = interp1d(x, peaks["pulses"][i], kind="cubic")
            new_peaks[i] = f(xnew)

        peaks["pulses"] = new_peaks
        n_samples = params.postprocessing.n_resamples

    # TODO: PARM FOR Flipping
    if params.postprocessing.enable_sign_correction:
        pulse_amplitude = peaks["pulses"][:, peaks["pulses"].shape[1] // 2]
        sign = np.ones_like(peaks["centers"])
        sign[pulse_amplitude < 0] = -1
        if params.postprocessing.polarity == "negative":
            sign = sign * -1
        peaks["pulses"] = peaks["pulses"] * sign[:, np.newaxis]

    if params.postprocessing.enable_centering:
        if params.postprocessing.centering_method in ["min", "max"]:
            for i, p in enumerate(peaks["pulses"]):
                peak_index = np.argmax(np.abs(p))
                diff_center = peak_index - p.shape[0] // 2
                center_peak = np.roll(p, diff_center)
                tukey_window = tukey(p.shape[0], 0.50)
                peaks["pulses"][i] = center_peak * tukey_window
        else:
            raise NotImplementedError()

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
