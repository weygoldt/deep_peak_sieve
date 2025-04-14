from typing import Annotated, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import labeled_comprehension
import seaborn as sns
import orjson
from pathlib import Path
import typer
from IPython import embed
from audioio.audioloader import AudioLoader

from deep_peak_sieve.utils.loggers import get_logger, configure_logging
from deep_peak_sieve.utils.datasets import Dict2Dataclass

app = typer.Typer()
log = get_logger(__name__)


def plot_peaks(
    raw_peak: np.ndarray,
    peak_archetype: np.ndarray,
    peak_start_on_raw,
    peak_end_on_raw,
    peak_start,
    peak_stop,
    peak_center,
    sample,
    labels,
):
    def on_press(event, index):
        if event.key == "n":
            log.info(f"Pulse {index} labeled as True")
            labels.append(True)
            plt.close()
        elif event.key == "t":
            log.info(f"Pulse {index} labeled as False")
            labels.append(False)
            plt.close()
        else:
            log.info(
                f"Key {event.key} not recognized. Use 'n' for True and 't' for False."
            )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        width_ratios=[3, 1],
        constrained_layout=True,
    )
    fig.canvas.mpl_connect("key_press_event", lambda event: on_press(event, sample))
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
    print(labels)


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the json samples file"),
    index: Annotated[Optional[int], typer.Option("--index", "-i")] = None,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
):
    configure_logging(verbose)

    labels = []
    # Load the data
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

        # Convert to dataclass for easier access
        data = Dict2Dataclass(data)

        # TODO: Handle different locations of raw data for each dtype (dataset type)
        # TODO: Save the labels, implement fixing them if failed

        for sample_indices, data_path in zip(data.sample_indices, data.files):
            data_path = Path(data_path)
            original_subdir = data_path.parent.stem.replace("_peaks", "")
            original_file = data_path.stem.replace("_peaks", "")
            original_path = (
                data_path.parent.parent / original_subdir / f"{original_file}.wav"
            )
            raw_file = AudioLoader(str(original_path))
            peak_file = Dict2Dataclass(np.load(data_path))

            for sample in sample_indices:
                peak_archetype = peak_file.peaks[sample]
                peak_start, peak_stop = peak_file.start_stop_index[sample]
                peak_center = peak_file.centers[sample]
                channels = peak_file.channels[sample]

                window_on_raw = peak_stop - peak_start
                peak_start_on_raw = peak_start - int(window_on_raw * 20)
                peak_end_on_raw = peak_stop + int(window_on_raw * 20)

                raw_peak = raw_file[peak_start_on_raw:peak_end_on_raw, :]

                # Plot the peaks
                plot_peaks(
                    raw_peak,
                    peak_archetype,
                    peak_start_on_raw,
                    peak_end_on_raw,
                    peak_start,
                    peak_stop,
                    peak_center,
                    sample,
                    labels,
                )
