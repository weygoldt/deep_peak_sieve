from typing import Annotated
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from IPython import embed
from pathlib import Path
import matplotlib.pyplot as plt
import typer
import orjson

from deep_peak_sieve.models.inception_time import InceptionTime


app = typer.Typer()


@app.command()
def main(
    path: Annotated[Path, typer.Argument(help="Path to the dataset file")],
):
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    all_labels = []
    all_peaks = []
    for sample_indices, data_path in zip(data["sample_indices"], data["files"]):
        data_path = Path(data_path)
        peak_file = np.load(data_path)

        labeled_peaks = peak_file["peaks"][sample_indices]
        labels = peak_file["labels"][sample_indices]

        all_labels.append(labels)
        all_peaks.append(labeled_peaks)

    labels = np.concatenate(all_labels)
    peaks = np.concatenate(all_peaks)

    print(f"Found {len(labels)} samples")

    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative integers")

    # add channel dimension to peaks
    peaks = np.expand_dims(peaks, axis=1)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        peaks, labels, stratify=labels, test_size=0.3
    )

    # Fit the model
    model = InceptionTime(
        x=x_train,
        y=y_train,
        filters=32,
        depth=6,
        models=5,
    )

    model.fit(
        learning_rate=0.001,
        batch_size=128,
        epochs=50,
        verbose=True,
    )

    # Evaluate the model
    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)
    print(
        "Training accuracy: {:.6f}".format(balanced_accuracy_score(y_train, yhat_train))
    )
    print("Test accuracy: {:.6f}".format(balanced_accuracy_score(y_test, yhat_test)))
