import json
import pathlib
from dataclasses import dataclass

import neo
import numpy as np
import numpy.typing as npt
from audioio import AudioLoader
from IPython import embed


@dataclass
class Metadata:
    samplerate: float
    channels: int
    duration: float
    frames: int


@dataclass
class Paths:
    data_path: str | pathlib.Path
    save_path: str | pathlib.Path
    layout_path: str | pathlib.Path


@dataclass
class SensoryArray:
    """Representation of the sensory array.

    Attributes
    ----------
    ids : Indiviudal ids of the sesory array
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
    data: AudioLoader | neo.OpenEphysBinaryIO
    metadata: Metadata
    paths: Paths
    sensoryarray: SensoryArray


def load_data(data_path, save_path, probe_path):
    wav_files = list(pathlib.Path(data_path).rglob("*.wav"))

    if wav_files:
        wav_files = [str(f) for f in wav_files]
        d = AudioLoader(wav_files)
        with open(probe_path) as f:
            seonsory_array = json.load(f)

        ids = np.arange(len(seonsory_array["coordinates"]))
        coordinates = np.array(seonsory_array["coordinates"])

        data_c = Data(
            d,
            Metadata(d.rate, d.channels, d.frames / d.rate, d.frames),
            Paths(data_path, save_path, probe_path),
            SensoryArray(
                ids,
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
            ),
        )
    else:
        dataset = neo.OpenEphysBinaryIO(data_path).read(lazy=True)
        data = dataset[0].segments[0].analogsignals[0]

        with pathlib.Path.open(probe_path) as f:
            seonsory_array = json.load(f)
        ids = np.array(seonsory_array["probes"][0]["device_channel_indices"])
        coordinates = np.array(
            seonsory_array["probes"][0]["contact_positions"]
        )
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
            SensoryArray(
                ids, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            ),
        )

    return data_c
