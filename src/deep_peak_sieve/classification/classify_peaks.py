import typer
from typing import Annotated
from pathlib import Path
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

from deep_peak_sieve.models.inception_time import InceptionTimeEnsemble
from deep_peak_sieve.utils.loggers import get_logger, configure_logging, get_progress
from deep_peak_sieve.utils.datasets import get_file_list


log = get_logger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    path: Annotated[Path, typer.Argument(help="Path to the dataset")],
    chkpt_dir: Annotated[
        Path, typer.Option(help="Path to the checkpoint directory")
    ] = Path("checkpoints"),
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
):
    configure_logging(verbose)
    model = InceptionTimeEnsemble()
    model.load_model(chkpt_dir)
    file_list, _, dtype = get_file_list(path, filetype="npz")

    # flatten file list if nested
    if isinstance(file_list[0], list):
        file_list = [item for sublist in file_list for item in sublist]

    with get_progress() as pbar:
        task = pbar.add_task("Classifying peaks", total=len(file_list))
        for i, file in enumerate(file_list):
            data = np.load(file)
            peaks = data["peaks"]

            # mu, std = np.mean(peaks), np.std(peaks)
            # peaks = (peaks - mu) / std
            # print(np.shape(peaks))
            # plt.plot(peaks.T, label="Peaks", color="grey", alpha=0.5)
            # plt.show()

            peaks = np.expand_dims(peaks, axis=1)  # Add channel dimension

            preds, probs = model.predict(peaks)

            new_data = {}
            for key, val in data.items():
                new_data[key] = val

            fig, axs = plt.subplots(1, 2, constrained_layout=True)
            peaks = peaks.squeeze()
            false_peaks = peaks[preds == 0, :]
            true_peaks = peaks[preds == 1, :]
            axs[0].plot(false_peaks.T, label="False Peaks", color="grey", alpha=0.5)
            axs[1].plot(true_peaks.T, label="True Peaks", color="grey", alpha=0.5)
            filename = f"classified_peaks_{i}.png"
            plt.savefig(filename)

            new_data["predicted_labels"] = preds
            new_data["predicted_probs"] = probs

            np.savez(file, **new_data)
            pbar.advance(task, 1)
