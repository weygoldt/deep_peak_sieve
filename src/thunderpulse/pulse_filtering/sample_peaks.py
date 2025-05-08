import gc
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Annotated

import nixio
import numpy as np
import orjson
import typer
from humanize.number import intword
from IPython import embed
from rich.prompt import Confirm

from thunderpulse.data_handling.data import get_file_list
from thunderpulse.nn.embedders import UmapEmbedder
from thunderpulse.utils.loggers import (
    configure_logging,
    get_logger,
    get_progress,
)

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

    # TODO: sample all flag
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
        # with get_progress() as pbar:
        #     task = pbar.add_task(
        #         "Collecting number of samples per file", total=len(self.files)
        #     )
        for file in self.files:
            if file.suffix == ".npz":
                data = np.load(file)
                key = list(data.keys())[0]
                num_samples_per_file.append(len(data[key]))
            elif file.suffix in [".nix", ".h5"]:
                with nixio.File.open(str(file), nixio.FileMode.ReadOnly) as f:
                    if len(f.blocks) == 0:
                        log.warning(f"File {file} has no blocks.")
                        continue
                    if len(f.blocks) > 1:
                        log.critical(
                            f"File {file} has more than one block. Using the first one."
                        )
                    block = f.blocks[0]
                    data_array = block.data_arrays["pulses"]
                    num_samples_per_file.append(data_array.shape[0])
            else:
                log.warning(
                    f"File {file} has unknown file type: {file.suffix}. Skipping."
                )
                # pbar.update(task, advance=1)

        num_samples_per_file = np.array(num_samples_per_file)
        log.info(
            f"Found {len(num_samples_per_file)} files with total of {intword(np.sum(num_samples_per_file))} peaks."
        )

        # Check if the total number of peaks fits the requested number of samples
        if np.sum(num_samples_per_file) < self.num_samples:
            raise ValueError(
                f"Not enough samples in the dataset. Found {np.sum(num_samples_per_file)}, but requested {self.num_samples}."
            )

        # Randomly sample peaks from each file to reach the requested number of samples
        total_frac_per_file = num_samples_per_file / np.sum(
            num_samples_per_file
        )
        target_samples_per_file = np.round(
            total_frac_per_file * self.num_samples
        ).astype(int)

        sample_indices = []
        with get_progress() as pbar:
            task = pbar.add_task(
                "Sampling peaks from each file", total=len(self.files)
            )
            for i in range(len(self.files)):
                num_samples = num_samples_per_file[i]
                num_samples_to_sample = target_samples_per_file[i]
                indices = np.sort(
                    np.random.randint(0, num_samples, num_samples_to_sample)
                )
                sample_indices.append(indices)
                pbar.update(task, advance=1)

        return sample_indices


def check_nixfile(f, name):
    passed = True
    if len(f.blocks) == 0:
        log.warning(f"File {name} has no blocks.")
        passed = False
    if len(f.blocks) > 1:
        log.critical(
            f"File {name} has more than one block. Using the first one."
        )
        passed = False
    return passed


@app.command()
def main(
    path: Path = typer.Argument(..., help="Path to the dataset"),
    num_samples: Annotated[
        int,
        typer.Option(
            "--num_samples", "-n", help="Number of total samples to draw"
        ),
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
        int,
        typer.Option("--verbose", "-v", count=True, help="Verbosity level"),
    ] = 0,
):
    configure_logging(verbose)
    data, _, dtype = get_file_list(
        path=path, filetype="h5", make_save_path=False
    )

    # check if data is nested list
    if isinstance(data[0], list):
        data = [item for sublist in data for item in sublist]

    log.info(f"Found {len(data)} files in the dataset.")

    sampler = StratifiedRandomSampler(data, num_samples=num_samples)

    log.info(f"Sampling {num_samples} peaks from {len(data)} files.")
    sample_indices = sampler.sample()

    savepath = None
    if dtype == "file" or dtype in ["dir", "subdir"]:
        savepath = path.parent / (path.stem + "_samples")
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")
    savepath = savepath.with_suffix(".json")

    sample_indices_index = [
        x for x in range(len(sample_indices)) if len(sample_indices[x]) > 0
    ]
    sample_indices = [sample_indices[i].tolist() for i in sample_indices_index]
    data = [data[i] for i in sample_indices_index]
    data = [str(file.resolve()) for file in data]
    #TODO: In Nix file

    json_data = {
        "dtype": dtype,
        "num_samples": num_samples,
        "sample_indices": sample_indices,
        "files": data,
    }

    overwrite = True
    if savepath.exists() and not force:
        log.info("Sample file already exists!")
        overwrite = Confirm.ask(
            "Do you want to overwrite the existing file?",
            default=False,
        )
    else:
        overwrite = True

    log.info("Collecting samples from each file.")
    outfile = savepath.with_suffix(".h5")

    if overwrite:
        outfile.unlink(missing_ok=True)
        sample_file = nixio.File(str(outfile), nixio.FileMode.Overwrite)
        sample_block = sample_file.create_block(
            name="samples", type_="samples"
        )
        json_file = savepath.with_suffix(".json")
        with open(json_file, "wb") as f:
            f.write(orjson.dumps(json_data))
    else:
        sys.exit()

    for i in range(len(sample_indices)):
        indices = np.sort(sample_indices[i])
        file = data[i]
        n = len(indices)
        files = [file] * n

        log.info(f"Collecting {n} samples from {file}.")
        if Path(file).suffix in [".nix", ".h5"]:
            with nixio.File.open(str(file), nixio.FileMode.ReadOnly) as f:
                if not check_nixfile(f, file):
                    msg = f"File {file} is not a valid NIX file. Exiting."
                    raise ValueError(msg)
                block = f.blocks[0]
                for arr in block.data_arrays:
                    name = arr.name
                    bindices = np.zeros(arr.shape[0], dtype=bool)
                    bindices[indices] = True
                    # TODO: This is super ugly but I did not find another working way
                    if len(arr.shape) == 1:
                        vals = arr[bindices]
                    elif len(arr.shape) == 2:
                        vals = arr[bindices, :]
                    elif len(arr.shape) == 3:
                        try:
                            vals = arr[bindices, :, :]
                        except Exception as e:
                            embed()
                            exit()
                    else:
                        msg = f"Array has unexpected shape: {arr.shape}"
                        raise ValueError(msg)

                    if i == 0:
                        data_array = sample_block.create_data_array(
                            name, "samples", data=vals
                        )
                    else:
                        data_array = sample_block.data_arrays[name]
                        data_array.append(vals)

        else:
            log.warning(
                f"File {file} has unknown file type: {Path(file).suffix}. Skipping."
            )
            continue
        gc.collect()

    # Close the file for now
    log.info("Extracted samples from all files, closing the file.")
    sample_file.close()

    # Fit the umap to the sampled data
    log.info("Re-opening the file for UMAP fitting.")
    dataset = nixio.File.open(
        str(outfile),
        nixio.FileMode.ReadWrite,
    )

    modelpath = outfile.parent
    pulses = dataset.blocks[0].data_arrays["mean_pulses"][:]
    embedder = UmapEmbedder("umap", str(modelpath / "umap.joblib"))
    log.info("Fitting UMAP to the sampled data.")
    embedder.fit(pulses)
    log.info("Predicting UMAP embedding for the sampled data.")
    yhat = embedder.predict(pulses)

    block = dataset.blocks[0]
    data_array = block.create_data_array(
        "umap_embedding", "embedding", data=yhat
    )
    dataset.close()
    log.info("UMAP embedding saved to the file.")
