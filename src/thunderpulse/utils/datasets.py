import numpy as np
from pathlib import Path
from audioio.audioloader import AudioLoader
from dataclasses import dataclass

from thunderpulse.utils.loggers import get_logger


log = get_logger(__name__)


@dataclass
class Dict2Dataclass:
    """Converts a dictionary to a dataclass."""

    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert the dataclass back to a dictionary."""
        return {key: value for key, value in self.__dict__.items()}


def get_file_list(path: Path, filetype="wav", make_save_path=True) -> tuple:
    """
    Discover the type of dataset based on the provided path.
    """

    file_list = []
    save_list = []

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")

    elif path.is_dir() and len(list(path.glob(f"*.{filetype}"))) > 0:
        file_list = sorted(list(path.glob(f"*.{filetype}")))
        save_dir = path.stem + "_peaks"
        save_path = path.parent / save_dir
        save_path.mkdir(exist_ok=True)
        save_file_names = [file.stem + "_peaks" for file in file_list]
        save_list = [save_path / name for name in save_file_names]
        return file_list, save_list, "dir"

    elif path.is_dir() and len(list(path.glob(f"*.{filetype}"))) == 0:
        subdirs = list(path.glob("*/"))
        save_dir = path.stem + "_peaks"
        save_path = path.parent / save_dir
        if make_save_path:
            save_path.mkdir(exist_ok=True)
        if len(subdirs) > 0:
            for subdir in subdirs:
                sub_file_list = sorted(list(subdir.glob(f"*.{filetype}")))
                if len(sub_file_list) > 0:
                    sub_save_dir = save_path / subdir.stem
                    if make_save_path:
                        sub_save_dir.mkdir(exist_ok=True)
                    file_list.append(sub_file_list)
                    save_file_names = [file.stem + "_peaks" for file in sub_file_list]
                    save_list.append([sub_save_dir / name for name in save_file_names])
        else:
            msg = f"Path {path} is a directory but contains no .wav files."
            raise ValueError(msg)
        return file_list, save_list, "subdir"

    elif path.is_file() and path.suffix == f".{filetype}":
        log.info("Dataset is a single file.")
        file_list = [path]
        save_name = path.stem + "_peaks.npy"
        save_list = [path.parent / save_name]
        return file_list, save_list, "file"
    else:
        msg = f"Path {path} is not a valid file or directory. Please provide a valid path."
        raise ValueError(msg)


def load_raw_data(path: Path, filetype="wav") -> tuple:
    """
    Load audio data from either a single file or all matching files in a directory.
    """

    file_list, save_list, dtype = get_file_list(path, filetype)
    if dtype == "file":
        # return [AudioLoader(str(path))], save_list
        return [str(path)], save_list

    elif dtype == "dir":
        # data = [AudioLoader(str(file)) for file in file_list]
        # return data, save_list
        return [str(file) for file in file_list], save_list
    elif dtype == "subdir":
        data = []
        savelist = []
        for subdir, savepaths in zip(file_list, save_list):
            for file, savefile in zip(subdir, savepaths):
                # sub_data = AudioLoader(str(file))
                # data.append(sub_data)
                data.append(str(file))
                savelist.append(savefile)
        return data, savelist
    else:
        msg = f"Path {path} is not a valid file or directory. Please provide a valid path."
        raise ValueError(msg)


def save_numpy(
    dataset: dict,
    savepath: Path,
):
    """
    Save the current dataset to disk, under a numbered filename (for chunked writing).
    """
    np.savez(savepath, **dataset)
