from os import makedev
from typing import Annotated
from abc import abstractmethod
import numpy as np
import typer
from pathlib import Path
import orjson
from rich.prompt import Confirm

from deep_peak_sieve.utils.loggers import get_logger, configure_logging
from deep_peak_sieve.utils.datasets import get_file_list

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)


"""
The ABC is a placeholder for more complicated sampling schemes. The current 
stratified random sampler is quite simple as it just draws random indices from each
file. But future plans include weighted sampling from peaks embedded in a hyperspace.
Already working witht the ABC makes sense here.
"""


class BaseSampler:
    """Base class for different peak samplers for labeling peaks."""

    def __init__(self, files: list, num_samples: int):
        self.files = files
        self.num_samples = num_samples

    @abstractmethod
    def sample(self) -> list:
        pass


class StratifiedRandomSampler(BaseSampler):
    """Randomly samples peaks from each file in the dataset."""

    def __init__(self, files: list, num_samples: int):
        super().__init__(files, num_samples)

    def sample(self) -> list:
        # Check how many peaks in each file
        num_samples_per_file = []
        for file in self.files:
            data = np.load(file)
            key = list(data.keys())[0]
            num_samples_per_file.append(len(data[key]))

        num_samples_per_file = np.array(num_samples_per_file)

        # Check if the total number of peaks fits the requested number of samples
        if np.sum(num_samples_per_file) < self.num_samples:
            raise ValueError(
                f"Not enough samples in the dataset. Found {np.sum(num_samples_per_file)}, but requested {self.num_samples}."
            )

        # Randomly sample peaks from each file to reach the requested number of samples
        total_frac_per_file = num_samples_per_file / np.sum(num_samples_per_file)
        target_samples_per_file = np.round(
            total_frac_per_file * self.num_samples
        ).astype(int)

        sample_indices = []
        for i in range(len(self.files)):
            num_samples = num_samples_per_file[i]
            num_samples_to_sample = target_samples_per_file[i]
            indices = np.random.randint(0, num_samples, num_samples_to_sample)
            sample_indices.append(indices)

        return sample_indices


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the dataset"),
    num_samples: Annotated[
        int, typer.Option("--num_samples", "-n", help="Number of total samples to draw")
    ] = 100,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force creating the samples.json file, overwriting existing files",
        ),
    ] = False,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="Verbosity level")
    ] = 0,
):
    configure_logging(verbose)
    data, _, dtype = get_file_list(path=path, filetype="npz", make_save_path=False)

    # check if data is nested list
    if isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]

    log.info(f"Found {len(data)} files in the dataset.")

    sampler = StratifiedRandomSampler(data, num_samples=num_samples)
    log.info(f"Sampling {num_samples} peaks from {len(data)} files.")

    sample_indices = sampler.sample()

    savepath = None
    if dtype == "file":
        savepath = path.parent / (path.stem + "_samples")
    elif dtype in ["dir", "subdir"]:
        savepath = path.parent / (path.stem + "_samples")
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")
    savepath = savepath.with_suffix(".json")

    sample_indices_index = [
        x for x in range(len(sample_indices)) if len(sample_indices[x]) > 0
    ]
    sample_indices = [sample_indices[i] for i in sample_indices_index]
    data = [data[i] for i in sample_indices_index]
    data = [str(file.resolve()) for file in data]
    json_data = {
        "dtype": dtype,
        "num_samples": num_samples,
        "sample_indices": sample_indices,
        "files": data,
    }

    if savepath.exists() and not force:
        log.info("Sample file already exists!")
        overwrite = Confirm.ask(
            "Do you want to overwrite the existing file?",
            default=False,
        )
    else:
        overwrite = True

    # Save the sample indices to a JSON file
    if overwrite:
        with open(savepath, "w") as f:
            f.write(
                orjson.dumps(json_data, option=orjson.OPT_SERIALIZE_NUMPY).decode(
                    "utf-8"
                )
            )
        log.info(f"Saved sample indices to {savepath.with_suffix('.json')}")
    else:
        log.info("Not overwriting the existing file.")
