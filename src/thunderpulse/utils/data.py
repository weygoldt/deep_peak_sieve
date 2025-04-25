import json
import pathlib
from dataclasses import dataclass

import neo
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
    channels: list[int]
    x: list[float]
    y: list[float]
    z: list[float]


@dataclass
class Data:
    data: AudioLoader | neo.OpenEphysBinaryIO
    metadata: Metadata
    paths: Paths
    sensoryarray: SensoryArray


def load_data(data_path, save_path, probe_path):
    wav_files = list(pathlib.Path(data_path).glob("**/*.wav"))

    if wav_files:
        wav_files = [str(f) for f in wav_files]
        d = AudioLoader(wav_files)
        try:
            seonsory_array =json.load(probe_path).read()
        except json.JSONDecodeError:
            print("Sensory Array cannot decoded as json")
        data_c = Data(
            data=d,
            metadata=Metadata(d.rate, d.channels, d.frames / d.rate, d.frames),
            paths=Paths(data_path, save_path, probe_path),
        )

    else:
        dataset = neo.OpenEphysBinaryIO(
            pathlib.Path(data_path).parent.parent
            / "neuronaldata"
            / "Recording-4"
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
