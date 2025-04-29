"""Main loop to parse large datasets and collect peaks."""

import json
from dataclasses import dataclass, asdict, field
import gc
from datetime import timedelta
from pathlib import Path
from typing import Annotated, List, Tuple, Optional, Dict, Any, Iterator

import humanize
import numpy as np
import typer
from audioio.audioloader import AudioLoader
from IPython import embed
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

from thunderpulse.data_handling.filters import bandpass_filter, notch_filter
from thunderpulse.data_handling.data import load_raw_data, save_numpy
from thunderpulse.utils.loggers import (
    configure_logging,
    get_logger,
    get_progress,
)

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)


# leaf-level parameter blocks
@dataclass(slots=True)
class FindPeaksKwargs:
    """Arguments forwarded to :func:`scipy.signal.find_peaks`."""

    height: float | np.ndarray | None = None
    threshold: float | np.ndarray | None = None
    distance: float | None = None
    prominence: float | np.ndarray | None = None
    width: float | np.ndarray | None = None

    # ----------------------------------------------------------------- helpers
    def to_kwargs(self, *, keep_none: bool = False) -> dict[str, Any]:
        """Convert to plain ``dict`` suitable for ``scipy.signal.find_peaks``.

        By default keys whose value is ``None`` are dropped.

        Examples
        --------
        >>> fp = FindPeaksKwargs(height=0.5, prominence=None)
        >>> fp.to_kwargs()
        {'height': 0.5}
        >>> signal.find_peaks(x, **fp.to_kwargs())
        """
        d = asdict(self)
        if not keep_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    # optional: make the object directly “expandable” via ** operator
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Allow ``dict(fp_kwargs)`` or ``**dict(fp_kwargs)``."""
        yield from self.to_kwargs().items()


@dataclass
class SavgolParameters:
    """Savitzky–Golay filter parameters."""

    window_length: int = 5
    polyorder: int = 3


@dataclass
class BandpassParameters:
    """IIR/FIR band-pass filter parameters."""

    lowcut: float = 0.1
    highcut: float = 3_000.0
    order: int = 5


@dataclass
class NotchParameters:
    """Notch filter (single frequency) parameters."""

    notch_freq: float = 50.0
    quality_factor: float = 30.0


# higher-level blocks
@dataclass
class PeakDetectionParameters:
    """Wrapper around *find_peaks* plus global peak-group criteria."""

    min_channels: int = 1
    mode: str = "both"  # 'pos', 'neg', 'both'
    peak_window: int = 10
    find_peaks_kwargs: FindPeaksKwargs = field(
        default_factory=lambda: FindPeaksKwargs(height=0.001)
    )


@dataclass
class FiltersParameters:
    """
    Arbitrary sequence of filters.

    *filters* is a list of **names**; *filter_params* holds a *parallel* list of
    parameter objects (must be the same length / order).
    """

    filters: list[str] = field(default_factory=lambda: ["savgol"])
    filter_params: list[object] = field(
        default_factory=lambda: [SavgolParameters()]
    )


@dataclass
class ResampleParameters:
    """Zero-hold / FFT resampling settings."""

    enabled: bool = True
    n_resamples: int = 512


# root config
@dataclass
class Params:
    """
    Full preprocessing configuration.

    Attributes
    ----------
    filters
        Definition & parameters of the DSP filter pipeline.
    peaks
        Peak detection settings.
    resample
        Resampling settings.
    """

    filters: FiltersParameters = field(default_factory=FiltersParameters)
    peaks: PeakDetectionParameters = field(
        default_factory=PeakDetectionParameters
    )
    resample: ResampleParameters = field(default_factory=ResampleParameters)

    # ── (de)serialisation helpers ──────────────────────────────────────
    def to_dict(self) -> dict:
        """Deep-convert to plain Python containers (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Re-build :class:`Params` from *asdict*-style dict."""
        # manual reconstruction because nested dataclasses are involved
        return cls(
            filters=FiltersParameters(
                filters=d["filters"]["filters"],
                filter_params=[
                    _build_filter_param(pname, pobj)
                    for pname, pobj in zip(
                        d["filters"]["filters"], d["filters"]["filter_params"]
                    )
                ],
            ),
            peaks=PeakDetectionParameters(
                min_channels=d["peaks"]["min_channels"],
                mode=d["peaks"]["mode"],
                peak_window=d["peaks"]["peak_window"],
                find_peaks_kwargs=FindPeaksKwargs(
                    **d["peaks"]["find_peaks_kwargs"]
                ),
            ),
            resample=ResampleParameters(**d["resample"]),
        )

    def to_json(self, **json_kwargs) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_json(cls, s: str) -> "Params":
        """De-serialise from JSON string."""
        return cls.from_dict(json.loads(s))


# utility to map filter-name → parameter-class
_FILTER_MAP = {
    "savgol": SavgolParameters,
    "bandpass": BandpassParameters,
    "notch": NotchParameters,
}


def _build_filter_param(name: str, payload: dict) -> object:
    """Construct the correct filter-param dataclass given its *name*."""
    cls = _FILTER_MAP.get(name)
    if cls is None:
        msg = f"Unknown filter type '{name}'."
        raise ValueError(msg)
    return cls(**payload)


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
    if len(params.filters) == 1:
        log.debug(f"Applying filter: {params.filters[0]}")
        return filter_methods[params.filters[0]](
            data,
            **params.filter_params[0],
            fs=rate,
        )
    # Apply all filters in sequence
    for filter_name, filter_params in zip(
        params.filters,
        params.filter_params,
        strict=True,
    ):
        log.debug(f"Applying filter: {filter_name}")
        data = filter_methods[filter_name](
            data,
            **filter_params,
            fs=rate,
        )
    return data


def detect_peaks(
    block: np.ndarray,
    mode: str = "both",
    find_peaks_kwargs: dict | None = None,
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

    peaks_list: list[np.ndarray] = []
    channels_list: list[np.ndarray] = []

    for ch in range(block.shape[1]):
        signal = block[:, ch]

        pos_peaks, _ = (
            find_peaks(
                signal,
                **(find_peaks_kwargs or {}),
            )
            if mode in {"peak", "both"}
            else (np.empty(0, dtype=int), None)
        )

        neg_peaks, _ = (
            find_peaks(
                -signal,
                **(find_peaks_kwargs or {}),
            )
            if mode in {"trough", "both"}
            else (np.empty(0, dtype=int), None)
        )

        # concatenate & sort to preserve chronological order
        peaks_ch = np.sort(np.concatenate((pos_peaks, neg_peaks)))
        peaks_list.append(peaks_ch)

        # build matching channel-id array
        channels_list.append(np.full_like(peaks_ch, ch, dtype=np.int32))

    return peaks_list, channels_list


def group_peaks_across_channels_by_time(
    peaks: list[np.ndarray],
    channels: list[np.ndarray],
    peak_window: int = 10,
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


def process_block(data: np.ndarray, rate: float, params: Params):
    # Ensure we treat mono data as [n_samples, 1]
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    # Apply filtering
    block_filtered = apply_filters(data, rate, params=params.filters)

    # Detect peaks on each channel
    peaks_list, channels_list = detect_peaks(
        block_filtered,
        mode=params.peaks.mode,
        find_peaks_kwargs=params.peaks.find_peaks_kwargs,
    )

    # Group them
    grouped_peaks, grouped_channels = group_peaks_across_channels_by_time(
        peaks_list,
        channels_list,
        # peak_window=int(min_peak_distance // 2),
        peak_window=params.peaks["peak_window"],
    )

    # Filter out groups that do not meet the threshold
    grouped_peaks, grouped_channels = filter_peak_groups(
        grouped_peaks, grouped_channels, params.peaks["min_channels"]
    )

    # Compute means of each group
    log.info(f"Found a total of {len(grouped_peaks)} peaks")
    if len(grouped_peaks) == 0:
        log.debug(f"Found 0 peaks. Skipping block {blockiterval} for now.")

    centers = [int(np.mean(g)) for g in grouped_peaks]

    # For each peak group, compute & store waveform
    for peakiterval, (pks, chans, center) in enumerate(
        zip(grouped_peaks, grouped_channels, centers, strict=False)
    ):
        chans = np.array(chans)
        if chans.dtype not in [np.int32, np.int64, np.bool]:
            # This should not happen, but just in case
            log.warning(
                f"Channel type is not int32 or int64 but {chans.dtype}. Converting to int64."
            )
            chans = np.array(chans, dtype=np.int64)

        pks = np.array(pks)
        if pks.dtype not in [np.int32, np.int64]:
            # This should not happen, but just in case
            log.warning(
                f"Peak type is not int32 or int64 but {pks.dtype}. Converting to int64."
            )
            pks = np.array(pks, dtype=np.int64)

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

        # print("-------------------------------------")
        # print("data.channels:")
        # print(data.channels)
        # print("pks:")
        # print(pks)
        # print("chans:")
        # print(chans)
        # print("block_filtered.shape:")
        # print(block_filtered.shape)

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
    pass


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

    if data.rate is None:
        raise ValueError("Data rate is None, cannot process file.")
    rate = data.rate

    blocksize = int(np.ceil(rate * buffersize_s))
    overlap = blocksize // 10

    # Configuration
    min_peak_distance = int(np.ceil(peak_distance_s * data.rate))
    around_peak_window = int(np.round(0.75 * min_peak_distance))
    window_length = int(np.ceil(smoothing_window_s * data.rate))
    polyorder = 3
    filtering_params = dict(
        mode="savgol",
        window_length=window_length
        if window_length > polyorder
        else polyorder + 1,
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

    dataset = initialize_dataset()
    collected_peak_counter = 0
    dataset_counter = 0

    num_blocks = int(np.ceil(data.shape[0] / (blocksize - overlap)))
    with get_progress() as pbar:
        desc = "Processing file"
        task = pbar.add_task(desc, total=num_blocks, transient=True)
        for blockiterval, block in enumerate(
            data.blocks(block_size=blocksize, noverlap=overlap)
        ):
            pbar.update(task, advance=1)

        dataset["peaks"] = np.array(dataset["peaks"])
        dataset["channels"] = np.array(dataset["channels"])
        dataset["amplitudes"] = np.array(dataset["amplitudes"])
        dataset["centers"] = np.array(dataset["centers"])
        dataset["start_stop_index"] = np.array(dataset["start_stop_index"])
        dataset["rate"] = data.rate

        save_numpy(dataset, save_path)
        gc.collect()


@app.command()
def main(
    datapath: Annotated[Path, typer.Argument(help="Path to the dataset.")],
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
    filetype: Annotated[
        str,
        typer.Option("--filetype", "-f", help="File type to detect peaks on."),
    ] = "wav",
    buffersize_seconds: Annotated[
        int, typer.Option("--buffersize", "-b", help="Buffer size in seconds.")
    ] = 60,
    min_channels_with_peaks: Annotated[
        int,
        typer.Option(
            "--min_channels", "-mc", help="Minimum channels with peaks."
        ),
    ] = 4,
    smoothing_window_s: Annotated[
        float,
        typer.Option(
            "--smoothing_window", "-sw", help="Smoothing window in seconds."
        ),
    ] = 0.0001,
    peak_distance_s: Annotated[
        float,
        typer.Option(
            "--peak_distance",
            "-pd",
            help="Minimum distance between peaks in seconds.",
        ),
    ] = 0.004,
    peak_height_threshold: Annotated[
        float,
        typer.Option(
            "--peak_height",
            "-ph",
            help="Minimum height of peaks in amplitude.",
        ),
    ] = 0.001,
    resample: Annotated[
        bool, typer.Option("--resample", "-r", help="Resample the data.")
    ] = True,
    n_resamples: Annotated[
        int, typer.Option("--n_resamples", "-nr", help="Number of resamples.")
    ] = 512,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", "-o", help="Overwrite existing files."),
    ] = False,
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
    file_list, save_list = load_raw_data(path=datapath, filetype=filetype)
    for data, save_path in zip(file_list, save_list, strict=False):
        # Skip if file already exists and overwrite is not set
        if save_path.with_suffix(".npz").exists() and not overwrite:
            log.info(f"File {save_path} already exists, skipping.")
            continue

        try:
            data = AudioLoader(data)
        except Exception as e:
            log.error(f"Failed to load {data}: {e}")
            continue
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
