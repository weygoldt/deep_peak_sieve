import gc
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from thunderpulse.models.inception_time import InceptionTimeEnsemble
from thunderpulse.utils.datasets import get_file_list
from thunderpulse.utils.loggers import (
    configure_logging,
    get_logger,
    get_progress,
)

log = get_logger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    path: Annotated[Path, typer.Argument(help="Path to the dataset")],
    chkpt_dir: Annotated[
        Path, typer.Option(help="Path to the checkpoint directory")
    ] = Path("checkpoints"),
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 128,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
):
    configure_logging(verbose)
    model = InceptionTimeEnsemble()
    model.load_model(chkpt_dir)
    file_list, _, dtype = get_file_list(
        path, filetype="npz", make_save_path=False
    )

    # flatten file list if nested
    if isinstance(file_list[0], list):
        file_list = [item for sublist in file_list for item in sublist]

    with get_progress() as pbar:
        task = pbar.add_task("Classifying peaks", total=len(file_list))
        for i, file in enumerate(file_list):
            try:
                data = np.load(file)
                peaks = data["peaks"]
            except Exception as e:
                log.error(f"Error loading file {file}: {e}")
                log.error(f"Skipping file {file}")
                continue

            if "predicted_labels" in data.keys():
                log.warning(f"File {file} already classified. Skipping.")
                continue

            # mu, std = np.mean(peaks), np.std(peaks)
            # peaks = (peaks - mu) / std
            # print(np.shape(peaks))
            # plt.plot(peaks.T, label="Peaks", color="grey", alpha=0.5)
            # plt.show()

            peaks = np.expand_dims(peaks, axis=1)  # Add channel dimension

            n_peaks = peaks.shape[0]
            if n_peaks == 0:
                log.warning(f"No peaks found in file {file}")
                continue

            n_batches = int(np.ceil(n_peaks / batch_size))
            if n_batches > 1:
                peaks_batched = np.array_split(peaks, n_batches)
            else:
                peaks_batched = [peaks]

            preds = []
            probs = []
            for j, batch in enumerate(peaks_batched):
                try:
                    batch_preds, batch_probs = model.predict(batch)
                except Exception as e:
                    log.error(f"Error predicting file {file}: {e}")
                    log.error(f"Skipping file {file}")
                    log.info(f"Shape of peaks: {batch.shape}")
                    continue

                preds.append(batch_preds)
                probs.append(batch_probs)

                torch.cuda.empty_cache()
                gc.collect()

            preds = np.concatenate(preds, axis=0)
            probs = np.concatenate(probs, axis=0)

            new_data = {}
            for key, val in data.items():
                new_data[key] = val

            # fig, axs = plt.subplots(1, 2, constrained_layout=True)
            # peaks = peaks.squeeze()
            # false_peaks = peaks[preds == 0, :]
            # true_peaks = peaks[preds == 1, :]
            # axs[0].plot(false_peaks.T, label="False Peaks", color="grey", alpha=0.5)
            # axs[1].plot(true_peaks.T, label="True Peaks", color="grey", alpha=0.5)
            # filename = f"classified_peaks_{i}.png"
            # plt.savefig(filename)

            new_data["predicted_labels"] = preds
            new_data["predicted_probs"] = probs

            np.savez(file, **new_data)
            pbar.advance(task, 1)
