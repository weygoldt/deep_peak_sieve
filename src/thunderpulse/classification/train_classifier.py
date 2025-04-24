import shutil
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import orjson
import seaborn as sns
import typer
from IPython import embed
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from thunderpulse.models.inception_time import InceptionTimeEnsemble
from thunderpulse.utils.loggers import configure_logging, get_logger

con = Console()
log = get_logger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    path: Annotated[
        Path, typer.Argument(help="Path to the dataset JSON file")
    ],
    n_models: Annotated[int, typer.Option("--n_models", "-n")] = 5,
    n_epochs: Annotated[int, typer.Option("--n_epochs", "-e")] = 20,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
):
    configure_logging(verbose)

    lightning_logdir = Path("lightning_logs")
    checkpoint_dir = Path("checkpoints")
    if lightning_logdir.exists():
        shutil.rmtree(lightning_logdir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    all_labels = []
    all_peaks = []
    log.info("Loading data ...")
    for sample_indices, data_path in zip(
        data["sample_indices"], data["files"], strict=False
    ):
        # patch data path to new mount point
        data_path = data_path.replace(
            "/mnt/data1/", "/home/weygoldt/mountpoint/"
        )
        # data_path = Path(data_path).resolve()

        data_path = Path(data_path)
        try:
            peak_file = np.load(data_path)
            labeled_peaks = peak_file["peaks"][sample_indices]
            labels = peak_file["labels"][sample_indices]
        except Exception as e:
            log.error(f"Failed to load {data_path}: {e}")
            continue

        all_labels.append(labels)
        all_peaks.append(labeled_peaks)

        log.info(f"Loaded {len(sample_indices)} samples from {data_path}")

    try:
        labels = np.concatenate(all_labels)
        peaks = np.concatenate(all_peaks)
    except:
        embed()
        exit()

    print(f"Found {len(labels)} samples")

    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative integers")

    # add channel dimension to peaks
    peaks = np.expand_dims(peaks, axis=1)

    # make a test split to test the full ensemble
    xtrain, xtest, ytrain, ytest = train_test_split(
        peaks, labels, stratify=labels, test_size=0.2, random_state=42
    )

    model = InceptionTimeEnsemble(
        n_models=n_models,
        input_size=peaks.shape[1],
        num_classes=np.unique(labels).shape[0],
        filters=32,
        depth=6,
    )

    # Fit and test the model
    model.fit(xtrain, ytrain, n_epochs)

    log.info(
        "Training complete. Testing the full ensemble on the test set ..."
    )

    # Load the best model from the checkpoints
    model.load_model(checkpoint_dir)

    # Do prediction on ensemble test set
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    error_color = sns.color_palette("deep")[1]
    correct_color = "grey"
    log.info(f"Evaluating ensemble on test set of {len(xtest)} samples")
    preds, probs = model.predict(xtest)

    acc = balanced_accuracy_score(ytest, preds)
    f1 = f1_score(ytest, preds, average="weighted")
    aps = average_precision_score(ytest, preds, average="weighted")
    log.info(
        f"Ensemble test set accuracy: {acc:.2f}, f1: {f1:.2f}, average precision: {aps:.2f}"
    )

    table = Table(
        title=f"Ensemble Test Set Results on {len(xtest)} samples",
        expand=False,
    )
    table.add_column("Metric", justify="center", style="cyan")
    table.add_column("Value", justify="center", style="green")
    table.add_row("Accuracy", f"{acc:.2f}")
    table.add_row("F1 Score", f"{f1:.2f}")
    table.add_row("Average Precision", f"{aps:.2f}")
    con.print(table)

    for yhat, y, x in zip(preds, ytest, xtest, strict=False):
        # x = np.expand_dims(x, axis=0)  # add channel dimension
        # yhat, probs = model.predict(x)
        color = correct_color if yhat == y else error_color
        label = "Correct" if yhat == y else "Error"
        ax = axs[0] if yhat == 1 else axs[1]
        x = x.squeeze()
        ax.plot(x, color=color, alpha=0.5, label=label)

    # make legend entries unique
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles, strict=False))
    axs[0].legend(unique_labels.values(), unique_labels.keys())
    axs[0].set_title("Label 1")
    axs[1].set_title("Label 0")
    plt.show()
