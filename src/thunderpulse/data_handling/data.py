from pathlib import Path
from dataclasses import dataclass

import neo
from audioio import AudioLoader
from IPython import embed
import numpy as np

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
class Data:
    """Composition of data and metadata."""

    data: AudioLoader | neo.OpenEphysBinaryIO
    metadata: Metadata
    paths: Paths


def load_data(data_path: Path, save_path: Path, probe_path: Path) -> Data:
    """Load OpenEphys or WAV data from the specified path."""
    wav_files = list(Path(data_path).rglob("*.wav"))

    if wav_files:
        file_list, _, dtype = get_file_list(
            data_path, "wav", make_save_path=False
        )[0]
        if isinstance(file_list[0], list):
            msg = (
                "Multiple directories found. Please provide a single "
                + "directory or file of a single recording session."
            )
            raise ValueError(msg)
        d = AudioLoader(file_list)
        data_c = Data(
            data=d,
            metadata=Metadata(d.rate, d.channels, d.frames / d.rate, d.frames),
            paths=Paths(data_path, save_path, probe_path),
        )

    else:
        dataset = neo.OpenEphysBinaryIO(
            Path(data_path).parent.parent / "neuronaldata" / "Recording-4"
        ).read(lazy=True)
        data = dataset[0].segments[0].analogsignals[0]

        data_c = Data(
            data,
            Metadata(
                data.sampling_rate.magnitude,
                data.shape[1],
                data.duration.magnitude,
                data.shape[0],
            ),
            Paths(data_path, save_path, probe_path),
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
