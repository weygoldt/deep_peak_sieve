from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import orjson
import seaborn as sns
import typer
from audioio.audioloader import AudioLoader

from thunderpulse.style import cm, set_dark_style
from thunderpulse.utils.loggers import configure_logging, get_logger

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)
# set_light_style()
set_dark_style()


# Disable built-in key mappings by setting them to empty lists.
matplotlib.rcParams["keymap.fullscreen"] = []
matplotlib.rcParams["keymap.home"] = []
matplotlib.rcParams["keymap.back"] = []
matplotlib.rcParams["keymap.forward"] = []
matplotlib.rcParams["keymap.pan"] = []
matplotlib.rcParams["keymap.zoom"] = []
matplotlib.rcParams["keymap.save"] = []
matplotlib.rcParams["keymap.quit"] = []
matplotlib.rcParams["keymap.grid"] = []
matplotlib.rcParams["toolbar"] = "None"


def plot_peaks(index, sample_indices, raw_file, peak_file):
    """
    Plot and label peak data from an audio file.

    Parameters
    ----------
    index : int
        Index of the sample to be plotted.
    sample_indices : list of int
        List containing all sample indices.
    raw_file : AudioLoader
        The audio loader object.
    peak_file : dict
        Dictionary containing peak information (peaks, centers, etc.).

    Returns
    -------
    str or None
        Pressed key label ('t' for True, 'f' for False, or 'c' to fix),
        or None if no recognized key was pressed.
    """
    keypress_var = None  # Local variable to store the key pressed.

    def on_press(event):
        nonlocal keypress_var
        valid_keys = {"t", "f", "c", "escape"}

        if event.key in valid_keys:
            # Log the recognized key press
            label_str = (
                "True"
                if event.key == "t"
                else "False"
                if event.key == "f"
                else "fix previous label"
                if event.key == "c"
                else "escape"
            )
            log.info(f"Pulse {index} labeled as {label_str}")
            keypress_var = event.key
            plt.close()
        else:
            log.info(
                f"Key '{event.key}' not recognized. "
                "Use 't' for True, 'f' for False, or 'c' to fix previous label, 'escape' to quit."
            )

    sample = sample_indices[index]
    peak_archetype = peak_file["peaks"][sample]
    peak_start, peak_stop = peak_file["start_stop_index"][sample]
    peak_center = peak_file["centers"][sample]

    # Expand the window around the peak for better context
    window_size = peak_stop - peak_start
    raw_start = peak_start - int(window_size * 5)
    raw_end = peak_stop + int(window_size * 5)

    # Extract data from the audio file
    raw_peak = raw_file[raw_start:raw_end, :]

    # Create figure and connect key-press event
    figsize = (40 * cm, 15 * cm)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        width_ratios=[3, 1],
    )
    fig.canvas.mpl_connect("key_press_event", on_press)

    # Left plot: multi-channel raw data with peak markings
    time_axis = np.linspace(raw_start, raw_end, raw_peak.shape[0])
    colors = sns.color_palette("Spectral", raw_peak.shape[1])
    for ch in range(raw_peak.shape[1]):
        axes[0].plot(
            time_axis,
            raw_peak[:, ch],
            label=f"Channel {ch}",
            alpha=1,
            color="white",
            lw=1,
        )

    axes[0].axvline(
        peak_center,
        linestyle="--",
        color="white",
        label="Peak Center",
        linewidth=0.75,
    )
    axes[0].axvline(
        peak_start,
        linestyle="--",
        color="grey",
        label="Peak Start",
        linewidth=0.75,
    )
    axes[0].axvline(
        peak_stop,
        linestyle="--",
        color="grey",
        label="Peak Stop",
        linewidth=0.75,
    )

    axes[0].set_xlabel("Time (samples)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw Data", loc="center")

    # Right plot: archetype (reference) for the peak
    xaxis = np.arange(peak_archetype.shape[0]) - peak_archetype.shape[0] // 2
    axes[1].plot(xaxis, peak_archetype, color="white", label="Peak Archetype")
    axes[1].set_xlabel("Interp. samples")
    axes[1].set_ylabel("Norm. amplitude")
    axes[1].set_title(f"Peak Archetype, sample {index}", loc="center")

    # Title with instructions
    fig.text(
        0.5,
        0.05,
        "Hint: Press 't' for True, 'f' for False, 'c' to fix, 'escape' to quit labeling safely",
        fontsize=10,
        fontweight="regular",
        horizontalalignment="center",
        verticalalignment="center",
        color="darkgrey",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.25)

    plt.show()
    return keypress_var


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the json samples file"),
    # index: Annotated[Optional[int], typer.Option("--index", "-i")] = None,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
):
    configure_logging(verbosity=verbose)

    # Load the data
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
        dtype = data["dtype"]
        log.info(
            f"Loaded data of type {dtype} with {data['num_samples']} samples."
        )

        # TODO: Make original data extension flexible (.wav)

    escape = False
    sample_counter = 0
    n_total_samples = [
        len(sample_indices) for sample_indices in data["sample_indices"]
    ]
    n_total_samples = sum(n_total_samples)
    for sample_indices, data_path in zip(
        data["sample_indices"], data["files"], strict=False
    ):
        log.info(f"Moving to new file {data_path}")
        data_path = Path(data_path)

        original_path = None
        if dtype == "dir":
            original_subdir = data_path.parent.stem.replace("_peaks", "")
            original_file = data_path.stem.replace("_peaks", "")
            original_path = (
                data_path.parent.parent
                / original_subdir
                / f"{original_file}.wav"
            )
        elif dtype == "subdir":
            original_dir = data_path.parent.parent.stem.replace("_peaks", "")
            original_subdir = data_path.parent.stem
            original_file = data_path.stem.replace("_peaks", "")
            original_path = (
                data_path.parent.parent.parent
                / original_dir
                / original_subdir
                / f"{original_file}.wav"
            )
            print(original_path)
        raw_file = AudioLoader(str(original_path))
        peak_file = np.load(data_path)
        if "labels" not in peak_file.files:
            log.info("No labels found in the peak file.")
            log.info("Creating new labels array.")
            labels = np.full(peak_file["peaks"].shape[0], -1, dtype=int)
        else:
            log.info("Labels found in the peak file.")
            log.info("Using existing labels array.")
            labels = peak_file["labels"]

        finished = False
        index = 0
        while not finished:
            sample_counter += 1
            log.info(f"Labeling sample {sample_counter} of {n_total_samples}")
            # Check if the index is already labeled
            if labels[sample_indices[index]] != -1:
                log.info(f"Sample {index} already labeled.")
                index += 1
                if index >= len(sample_indices):
                    log.info("Reached the end of the samples for this file.")
                    finished = True
                continue

            # Plot the peaks
            key = plot_peaks(
                index=index,
                sample_indices=sample_indices,
                raw_file=raw_file,
                peak_file=peak_file,
            )

            log.debug(f"Key pressed: {key}")
            if key == "escape":
                log.info("Exiting labeling process.")
                finished = True
                escape = True
                break

            label = -1
            if key == "c":
                index = index - 1
                if index >= 0:
                    log.info(f"Correcting label for sample {index}")
                    key = plot_peaks(
                        index=index,
                        sample_indices=sample_indices,
                        raw_file=raw_file,
                        peak_file=peak_file,
                    )
                    if key == "t":
                        label = 1
                    elif key == "f":
                        label = 0
                    else:
                        raise ValueError(
                            f"Invalid key pressed for correction: {key}"
                        )
                else:
                    index = 0
                    log.info("No previous sample to correct.")
                    continue
            elif key == "t":
                label = 1
            elif key == "f":
                label = 0
            else:
                raise ValueError(
                    f"Invalid key pressed: {key}. Expected 't', 'f', or 'c'."
                )

            # Save the label
            labels[sample_indices[index]] = label

            index += 1
            if index >= len(sample_indices):
                log.info("Reached the end of the samples for this file.")
                finished = True

        # Save the labels
        new_peak_file = {
            "labels": labels,
        }
        for key, value in peak_file.items():
            if key not in new_peak_file:
                new_peak_file[key] = value

        np.savez(data_path, **new_peak_file)
        log.debug(f"Labels added to extracted peask at {data_path}.")
        if escape:
            break
