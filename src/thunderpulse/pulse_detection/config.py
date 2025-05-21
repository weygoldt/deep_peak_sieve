"""Objects to serialize and deserialize detector configuration."""

import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
import orjson
from dacite import Config, from_dict
from IPython import embed

from thunderpulse.data_handling.data import SensorArray
from thunderpulse.dsp.filters import (
    bandpass_filter,
    notch_filter,
    savitzky_golay_filter,
)

_INDENT = 2

CenteringMethod = Literal["max", "min", "max_diff"]
PulsePolarity = Literal["positive", "negative"]


@dataclass(slots=True)
class KwargsDataclass:
    """Base class for kwargs dataclasses.

    This is used to enforce that all subclasses are "slots" dataclasses.
    """

    def __post_init__(self) -> None:
        """Check that all fields are defined as slots."""
        if not is_dataclass(self):
            raise TypeError("KwargsDataclass must be a dataclass.")
        if not hasattr(self, "__slots__"):
            raise TypeError("KwargsDataclass must have __slots__ defined.")

    def to_kwargs(self, *, keep_none: bool = False) -> dict[str, Any]:
        """Convert to plain ``dict`` suitable for ``scipy.signal.find_peaks``.

        By default keys whose value is ``None`` are dropped.
        """
        d = asdict(self)
        if not keep_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Allow ``dict(fp_kwargs)`` or ``**dict(fp_kwargs)``."""
        yield from self.to_kwargs().items()


# leaf-level parameter blocks
@dataclass(slots=True)
class FindPeaksKwargs(KwargsDataclass):
    """Arguments forwarded to :func:`scipy.signal.find_peaks`."""

    # TODO: This should be in seconds and then converted to samples
    # once the samplerate is known

    height: float | np.ndarray | None = None
    threshold: float | np.ndarray | None = None
    distance: float | None = None
    prominence: float | np.ndarray | None = 0.001
    width: float | np.ndarray | None = None


@dataclass(slots=True)
class PreProcessingParameters(KwargsDataclass):
    """Apply Prefilter operations."""

    common_median_reference: bool = False


@dataclass(slots=True)
class SavgolParameters(KwargsDataclass):
    """Savitzky–Golay filter parameters."""

    window_length_s: float | None = 0.0005  # seconds
    polyorder: int | None = 3


@dataclass
class BandpassParameters(KwargsDataclass):
    """Band-pass filter parameters."""

    lowcut: float | None = 0.1
    highcut: float | None = 3_000.0
    order: int | None = 5


@dataclass
class NotchParameters(KwargsDataclass):
    """Notch filter (single frequency) parameters."""

    notch_freq: float | None = 50.0
    quality_factor: float | None = 30.0


# higher-level blocks
@dataclass
class PeakDetectionParameters(KwargsDataclass):
    """Wrapper around *find_peaks* plus global peak-group criteria."""

    min_channels: int = 1
    mode: str = "both"  # 'peak', 'trough', 'both'
    min_peak_distance_s: float = 0.001  # seconds
    cutout_window_around_peak_s: float = 0.005  # seconds
    take_pulse_with_max_amplitude: bool = False
    distance_channels: float = 50  # distance unit sensoryarray

    find_peaks_kwargs: FindPeaksKwargs = field(
        default_factory=lambda: FindPeaksKwargs(height=0.001)
    )


# @dataclass
# class FiltersParameters(KwargsDataclass):
#     """
#     Arbitrary sequence of filters.
#
#     *filters* is a list of **names**; *filter_params* holds a *parallel* list of
#     parameter objects (must be the same length / order).
#     """
#     filters: list[str] = field(
#         default_factory=lambda: ["savgol", "bandpass", "notch"]
#     )
#     filter_params: list[KwargsDataclass] = field(
#         default_factory=lambda: [
#             SavgolParameters(),
#             BandpassParameters(),
#             NotchParameters(),
#         ]
#     )


@dataclass
class FiltersParameters(KwargsDataclass):
    """
    Arbitrary sequence of filters.

    *filters* is a list of **names**; *filter_params* holds a *parallel* list of
    parameter objects (must be the same length / order).
    """

    savgol: Optional[SavgolParameters] = field(
        default_factory=SavgolParameters
    )
    bandpass: Optional[BandpassParameters] = field(
        default_factory=BandpassParameters
    )
    notch: Optional[NotchParameters] = field(default_factory=NotchParameters)

    filter_ordering: list[str] = field(
        default_factory=lambda: ["savgol", "bandpass", "notch"]
    )

    def remove_filters_with_all_none_params(self):
        def all_params_none(obj):
            if obj is None:
                return True
            if hasattr(obj, "__dataclass_fields__"):
                return all(
                    getattr(obj, f) is None for f in obj.__dataclass_fields__
                )
            if isinstance(obj, dict):
                return all(v is None for v in obj.values())
            return False

        for filter_name in list(self.filter_ordering):
            value = getattr(self, filter_name, None)
            if all_params_none(value):
                setattr(self, filter_name, None)
        self.filter_ordering = [
            name
            for name in self.filter_ordering
            if getattr(self, name, None) is not None
        ]


@dataclass
class PostProcessingParameters(KwargsDataclass):
    enable_centering: bool = True
    enable_resampling: bool = True
    enable_sign_correction: bool = True
    enable_normalization: bool = True
    n_resamples: int | None = 512
    centering_method: CenteringMethod | None = "max"
    polarity: PulsePolarity | None = "positive"


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

    preprocessing: PreProcessingParameters = field(
        default_factory=PreProcessingParameters
    )
    filters: FiltersParameters = field(default_factory=FiltersParameters)
    peaks: PeakDetectionParameters = field(
        default_factory=PeakDetectionParameters
    )
    postprocessing: PostProcessingParameters = field(
        default_factory=PostProcessingParameters
    )
    sensoryarray: SensorArray = field(default_factory=SensorArray)
    buffersize_s: float = 60.0  # seconds

    # ── (de)serialisation     helpers ──────────────────────────────────────
    def to_dict(self) -> dict:
        """Deep-convert to plain Python containers (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Re-build :class:`Params` from *asdict*-style dict."""
        # manual reconstruction because nested dataclasses are involved
        for key in d["sensoryarray"]:
            if hasattr(d["sensoryarray"][key], "tolist"):
                d["sensoryarray"][key] = d["sensoryarray"][key].tolist()
        params = from_dict(
            data_class=cls,
            data=d,
        )
        params.sensoryarray.to_numpy()
        return params

    def to_json(self, **json_kwargs) -> str:
        """Serialise to JSON string."""
        return orjson.dumps(self.to_dict(), option=orjson.OPT_SERIALIZE_NUMPY)

    @classmethod
    def from_json(cls, s: str) -> "Params":
        """De-serialise from JSON string."""
        with open(s) as f:
            json_file = json.loads(f.read())

        return cls.from_dict(json_file)


# utility to map filter-name → parameter-class
filter_map = {
    "savgol": savitzky_golay_filter,
    "bandpass": bandpass_filter,
    "notch": notch_filter,
}

filter_config_map = {
    "savgol": SavgolParameters,
    "bandpass": BandpassParameters,
    "notch": NotchParameters,
}


def _build_filter_param(name: str, payload: dict) -> object:
    """Construct the correct filter-param dataclass given its *name*."""
    cls = filter_config_map.get(name)
    if cls is None:
        msg = f"Unknown filter type '{name}'."
        raise ValueError(msg)
    return cls(**payload)


def _fmt_scalar(x: Any) -> str:
    if isinstance(x, (Path, str)):
        return f'"{x}"'
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, np.ndarray):
        if x.size > 10:  # large array → just shape
            return f"<ndarray shape={x.shape} dtype={x.dtype}>"
        return np.array2string(x, separator=",", precision=3)
    return str(x)


def _pretty(obj: Any, level: int) -> str:
    pad = " " * (_INDENT * level)

    if is_dataclass(obj):
        cls_name = obj.__class__.__name__
        lines = [f"{pad}{cls_name}:"]
        for f in fields(obj):
            val = getattr(obj, f.name)
            child = _pretty(val, level + 1)
            if isinstance(val, (list, dict)) or is_dataclass(val):
                lines.append(f"{pad}{' ' * _INDENT}{f.name}:")
                lines.append(child)
            else:
                lines.append(
                    f"{pad}{' ' * _INDENT}{f.name}: {_fmt_scalar(val)}"
                )
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        if not obj:
            return f"{pad}[]"
        lines = []
        for item in obj:
            child = _pretty(item, level + 1)
            bullet = "-" if not isinstance(item, (list, dict)) else ""
            if bullet:
                lines.append(f"{pad}{bullet} {_fmt_scalar(item)}")
            else:
                lines.append(f"{pad}-")
                lines.append(child)
        return "\n".join(lines)

    if isinstance(obj, dict):
        if not obj:
            return f"{pad}{{}}"
        lines = []
        for k, v in obj.items():
            child = _pretty(v, level + 1)
            if isinstance(v, (list, dict)) or is_dataclass(v):
                lines.append(f"{pad}{k}:")
                lines.append(child)
            else:
                lines.append(f"{pad}{k}: {_fmt_scalar(v)}")
        return "\n".join(lines)

    return f"{pad}{_fmt_scalar(obj)}"


def pretty_print_config(cfg: Any, *, file=sys.stdout) -> None:
    """
    Nicely display a nested dataclass configuration in the shell.

    Parameters
    ----------
    cfg
        Instance of your root configuration dataclass (e.g. ``Params()``).
    file
        Stream to write to (defaults to ``sys.stdout``).
    """
    print(_pretty(cfg, 0), file=file)


if __name__ == "__main__":
    # Example usage
    cfg = Params()
    pretty_print_config(cfg)

    with open("config.json", "w") as f:
        jsonstr = cfg.to_json(indent=2, sort_keys=True)
        f.write(jsonstr)
