import sys
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import nixio
import numpy as np
import typer
from humanize.number import intword
from IPython import embed
from nixio.exceptions import DuplicateName
from rich.prompt import Confirm

from thunderpulse.data_handling.data import get_file_list
from thunderpulse.nn.embedders import UmapEmbedder
from thunderpulse.utils.loggers import (
    configure_logging,
    get_logger,
)

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)

np.random.seed(42)


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the dataset"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force creating the samples.json file, overwriting existing files",
        ),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Verbosity level"),
    ] = 0,
):
    configure_logging(verbose)
    data, _, dtype = get_file_list(
        path=path, filetype="nix", make_save_path=False
    )

    # check if data is nested list
    if isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]

    data = list(Path(path).rglob("*.nix"))
    nix_file = nixio.File.open(str(data[0]), nixio.FileMode.ReadWrite)

    block = nix_file.blocks[0]
    data_arrays = block.data_arrays
    pulses = data_arrays["pulses"][:]
    channels = data_arrays["channels"][:]
    pulse_min = data_arrays["prominent_pulses"][:]
    pulses_selection = pulses[pulse_min]
    selectes_pulses = channels[pulse_min]

    unique_channels = np.unique(channels)
    for ch in unique_channels:
        log.info(f"Channel {ch}")
        channel_index = selectes_pulses == ch
        pulses_channel = pulses_selection[channel_index, :]
        embedder = UmapEmbedder(
            "umap", str(data[0].parent / f"umap_channel_{ch}.joblib")
        )
        log.info("Fitting UMAP to the data.")

        embedder.fit(pulses_channel)
        log.info("Predicting UMAP embedding for the sampled data.")
        yhat = embedder.predict(pulses_channel)
        try:
            data_array = block.create_data_array(
                f"umap_embedding_channel_{ch}", "embedding", data=yhat
            )
        except DuplicateName:
            overwrite = True
            if not force:
                overwrite = Confirm.ask(
                    "Do you want to overwrite the existing file?",
                    default=False,
                )
            if overwrite:
                del data_arrays[f"umap_embedding_channel_{ch}"]
                data_array = block.create_data_array(
                    f"umap_embedding_channel_{ch}", "embedding", data=yhat
                )
            else:
                log.info("Exiting Umap embedding")
                nix_file.close()
                sys.exit(1)
    nix_file.close()

    # hdb = HDBSCAN(min_cluster_size=20, n_jobs=-1)
    # labels = hdb.fit_predict(yhat)
    # labels_offset = np.where(
    #     labels != -1, labels + global_label_offset, -1
    # )
    # all_labels[channel_index] = labels_offset
    # # Update offset for next channel
    # if labels.max() != -1:
    #     global_label_offset += labels.max() + 1

    # unique_labels = np.unique(labels)
    # colors = plt.cm.get_cmap("tab10", len(unique_labels))
    # for idx, label in enumerate(unique_labels):
    #     mask = labels == label
    #     if label == -1:
    #         # Noise points
    #         plt.scatter(
    #             yhat[mask, 0],
    #             yhat[mask, 1],
    #             c="k",
    #             marker="x",
    #             label="Noise",
    #             s=10,
    #         )
    #     else:
    #         plt.scatter(
    #             yhat[mask, 0],
    #             yhat[mask, 1],
    #             color=colors(idx),
    #             s=10,
    #             label=f"Cluster {label}",
    #         )
    # plt.xlabel("Embedding 1")
    # plt.ylabel("Embedding 2")
    # plt.title("HDBSCAN Clustering")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
