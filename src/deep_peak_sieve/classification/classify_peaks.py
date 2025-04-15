from typing import Annotated
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from IPython import embed
from pathlib import Path
import matplotlib.pyplot as plt
import typer
import orjson
import seaborn as sns

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    n_samples = len(x_test)
    n_classes = len(np.unique(labels))
    correct_color = "grey"
    incorrect_color = "red"

    colors = sns.color_palette("deep", n_colors=n_classes)
    print(np.unique(yhat_test))
    for i in range(n_classes):
        yhat = yhat_test[y_test == i]
        print(f"Class {i}: {len(yhat)} samples")
        sns.kdeplot(
            yhat,
            label=f"Class {i}",
            ax=ax,
            color=colors[i],
        )

    plt.show()
    exit()

    thresh = 0.5

    for i in range(n_samples):
        label = y_test[i]
        prediction = yhat_test[i] > thresh
        correct = label == prediction
        if label == 1:
            color = correct_color if correct else incorrect_color
            ax[0].plot(
                x_test[i].flatten(),
                color=color,
                alpha=0.5,
            )
        else:
            color = incorrect_color if correct else correct_color
            ax[1].plot(
                x_test[i].flatten(),
                color=color,
                alpha=0.5,
            )
    plt.show()
