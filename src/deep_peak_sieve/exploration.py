from typing import Annotated
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import typer
from audioio.audioloader import AudioLoader

from deep_peak_sieve.utils.datasets import Dict2Dataclass

app = typer.Typer()


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the numpy file"),
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
):
    np_files = sorted(list(path.glob("*.npz")))
    print(np_files)

    for file in np_files:
        original_subdir = file.parent.stem.replace("_peaks", "")
        original_file = file.stem.replace("_peaks", "")
        original_path = file.parent.parent / original_subdir / f"{original_file}.wav"
        raw_file = AudioLoader(str(original_path))
        peak_file = Dict2Dataclass(np.load(file))

        for sample in range(len(peak_file.peaks)):
            peak_archetype = peak_file.peaks[sample]
            peak_start, peak_stop = peak_file.start_stop_index[sample]
            peak_center = peak_file.centers[sample]
            channels = peak_file.channels[sample]

            window_on_raw = peak_stop - peak_start
            peak_start_on_raw = peak_start - int(window_on_raw * 20)
            peak_end_on_raw = peak_stop + int(window_on_raw * 20)

            raw_peak = raw_file[peak_start_on_raw:peak_end_on_raw, :]

            fig, ax = plt.subplots(
                1,
                2,
                figsize=(15, 6),
                width_ratios=[3, 1],
                constrained_layout=True,
            )
            x = np.linspace(
                peak_start_on_raw,
                peak_end_on_raw,
                raw_peak.shape[0],
            )
            for ch in range(raw_peak.shape[1]):
                ax[0].plot(x, raw_peak[:, ch], label=f"Channel {ch}")

            ax[0].axvline(
                peak_center,
                color="red",
                linestyle="--",
                label="Peak Center",
            )
            ax[0].axvspan(
                peak_start,
                peak_stop,
                color="green",
                alpha=0.5,
                label="Peak Window",
            )

            x = np.arange(peak_archetype.shape[0])
            ax[1].plot(x, peak_archetype, label="Peak Archetype", color="k")
            plt.show()
