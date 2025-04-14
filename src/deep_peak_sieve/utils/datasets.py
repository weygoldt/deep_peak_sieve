import numpy as np
from pathlib import Path
from audioio.audioloader import AudioLoader

from deep_peak_sieve.utils.loggers import get_logger

log = get_logger(__name__)


def get_file_list(path: Path, filetype="wav") -> tuple:
    """
    Discover the type of dataset based on the provided path.
    """

    file_list = []

    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    elif path.is_dir() and len(list(path.glob("*.wav"))) > 0:
        file_list = sorted(list(path.glob("*.wav")))
        return file_list, "dir"
    elif path.is_dir() and len(list(path.glob("*.wav"))) == 0:
        subdirs = list(path.glob("*/"))
        if len(subdirs) > 0:
            for subdir in subdirs:
                sub_file_list = sorted(list(subdir.glob("*.wav")))
                if len(sub_file_list) > 0:
                    file_list.extend(sub_file_list)
        return file_list, "subdir"
    elif path.is_file() and path.suffix == ".wav":
        log.info("Dataset is a single file.")
        file_list = [path]
        return file_list, "file"
    else:
        msg = f"Path {path} is not a valid file or directory. Please provide a valid path."
        raise ValueError(msg)


def load_data(path: Path, filetype="wav") -> list:
    """
    Load audio data from either a single file or all matching files in a directory.
    """

    file_list, dtype = get_file_list(path, filetype)
    if dtype == "file":
        return [AudioLoader(str(path))]
    elif dtype == "dir":
        return [AudioLoader(file_list)]
    elif dtype == "subdir":
        data = []
        for subdir in file_list:
            sub_data = AudioLoader(str(subdir))
            data.append(sub_data)
        return data
    else:
        msg = f"Path {path} is not a valid file or directory. Please provide a valid path."
        raise ValueError(msg)
