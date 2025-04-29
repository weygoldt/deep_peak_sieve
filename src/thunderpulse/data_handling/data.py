import json
from dataclasses import dataclass
from pathlib import Path

import neo
import numpy as np
import numpy.typing as npt
from audioio import AudioLoader
from IPython import embed

from thunderpulse.utils.loggers import get_logger

log = get_logger(__name__)


@dataclass
class Metadata:
    """Wrapper for metadata."""

    samplerate: float
    channels: int
    duration: float
    frames: int


@dataclass
class Paths:
    """Wrapper for paths."""

    data_path: str | Path
    save_path: str | Path
    layout_path: str | Path


@dataclass
class SensorArray:
    """Representation of the sensor array.

    Attributes
    ----------
    ids : Indiviudal ids of the sesor array
    x : x-positions
    y : y-positions
    z : z-positions
    """

    ids: npt.NDArray[np.int16]
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    z: npt.NDArray[np.float32]


@dataclass
class Data:
    """Composition of data and metadata."""

    data: AudioLoader | neo.OpenEphysBinaryIO
    metadata: Metadata
    paths: Paths
    sensorarray: SensorArray


def load_data(
    data_path: Path | str, save_path: Path | str, probe_path: Path | str
) -> Data:
    """Load OpenEphys or WAV data from the specified path."""
    data_path = Path(data_path)
    save_path = Path(save_path)
    probe_path = Path(probe_path)
    wav_files = list(Path(data_path).rglob("*.wav"))

    if len(wav_files) > 0:
        log.debug("Data directory has wav files")
        with Path.open(probe_path) as f:
            seonsory_array = json.load(f)

        file_list, _, dtype = get_file_list(
            data_path, "wav", make_save_path=False
        )
        log.debug(f"File list: {file_list}")
        if isinstance(file_list[0], list):
            msg = (
                "Multiple directories found. Please provide a single "
                + "directory or file of a single recording session."
            )
            raise ValueError(msg)
        if isinstance(file_list[0], Path):
            file_list = [str(file) for file in file_list]

        d = AudioLoader(file_list)
        ids = np.arange(len(seonsory_array["coordinates"]))
        coordinates = np.array(seonsory_array["coordinates"])

        data_c = Data(
            d,
            Metadata(d.rate, d.channels, d.frames / d.rate, d.frames),
            Paths(data_path, save_path, probe_path),
            SensorArray(
                ids,
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
            ),
        )
    else:
        dataset = neo.OpenEphysBinaryIO(data_path).read(lazy=True)
        data = dataset[0].segments[0].analogsignals[0]

        with Path.open(probe_path) as f:
            sensor_array = json.load(f)
        ids = np.array(sensor_array["probes"][0]["device_channel_indices"])
        coordinates = np.array(sensor_array["probes"][0]["contact_positions"])
        if coordinates.shape[1] != 3:
            coordinates = np.hstack(
                (coordinates, np.zeros_like(coordinates[:, 0]).reshape(-1, 1))
            )

        data_c = Data(
            data,
            Metadata(
                data.sampling_rate.magnitude,
                data.shape[1],
                data.duration.magnitude,
                data.shape[0],
            ),
            Paths(data_path, save_path, probe_path),
            SensorArray(
                ids, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            ),
        )

    return data_c


def get_file_list(
    path: Path, filetype: str = "wav", make_save_path: bool = True
) -> tuple:
    """Discover the type of WAV dataset based on the provided path."""
    file_list = []
    save_list = []

    if not path.exists():
        raise FileNotFoundError()

    if path.is_dir() and len(list(path.glob(f"*.{filetype}"))) > 0:
        file_list = sorted(path.glob(f"*.{filetype}"))
        save_dir = path.stem + "_peaks"
        save_path = path.parent / save_dir
        save_path.mkdir(exist_ok=True)
        save_file_names = [file.stem + "_peaks" for file in file_list]
        save_list = [save_path / name for name in save_file_names]
        return file_list, save_list, "dir"

    if path.is_dir() and len(list(path.glob(f"*.{filetype}"))) == 0:
        subdirs = list(path.glob("*/"))
        save_dir = path.stem + "_peaks"
        save_path = path.parent / save_dir
        if make_save_path:
            save_path.mkdir(exist_ok=True)
        if len(subdirs) > 0:
            for subdir in subdirs:
                sub_file_list = sorted(subdir.glob(f"*.{filetype}"))
                if len(sub_file_list) > 0:
                    sub_save_dir = save_path / subdir.stem
                    if make_save_path:
                        sub_save_dir.mkdir(exist_ok=True)
                    file_list.append(sub_file_list)
                    save_file_names = [
                        file.stem + "_peaks" for file in sub_file_list
                    ]
                    save_list.append(
                        [sub_save_dir / name for name in save_file_names]
                    )
        else:
            msg = f"Path {path} is a directory but contains no .wav files."
            raise ValueError(msg)
        return file_list, save_list, "subdir"

    if path.is_file() and path.suffix == f".{filetype}":
        log.info("Dataset is a single file.")
        file_list = [path]
        save_name = path.stem + "_peaks.npy"
        save_list = [path.parent / save_name]
        return file_list, save_list, "file"
    msg = (
        f"Path {path} is not a valid file or directory. "
        + "Please provide a valid path."
    )
    raise ValueError(msg)


def load_raw_data(path: Path, filetype: str = "wav") -> tuple:
    """Load audio data from single or multiple files in a directory."""
    file_list, save_list, dtype = get_file_list(path, filetype)
    if dtype == "file":
        return [str(path)], save_list
    if dtype == "dir":
        return [str(file) for file in file_list], save_list
    if dtype == "subdir":
        data = []
        savelist = []
        for subdir, savepaths in zip(file_list, save_list, strict=False):
            for file, savefile in zip(subdir, savepaths, strict=False):
                data.append(str(file))
                savelist.append(savefile)
        return data, savelist
    msg = (
        f"Path {path} is not a valid file or directory. "
        + "Please provide a valid path."
    )
    raise ValueError(msg)


def save_numpy(
    dataset: dict,
    savepath: Path,
) -> None:
    """Save a numpy dataset to disk."""
    np.savez(savepath, **dataset)
