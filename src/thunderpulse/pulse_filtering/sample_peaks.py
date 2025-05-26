import gc
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Annotated, Sequence

import matplotlib.pyplot as plt
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

np.random.seed(42)


"""
The ABC is a placeholder for more complicated sampling schemes. The current 
stratified random sampler is quite simple as it just draws random indices from each
file. But future plans include weighted sampling from peaks embedded in a hyperspace.
Already working witht the ABC makes sense here.
"""


class BaseSampler:
    """Base class for different peak samplers for labeling peaks."""

    # TODO: sample all flag
    def __init__(
        self, files: list, num_samples: int, sample_all: bool = False
    ):
        self.files = files
        self.num_samples = num_samples
        self.sample_all = sample_all

    @abstractmethod
    def sample_all_files(self) -> list:
        pass

    @abstractmethod
    def sample_single_file(self, filename) -> list:
        pass


class StratifiedRandomSampler(BaseSampler):
    """Randomly samples peaks from each file in the dataset."""

    def __init__(
        self,
        files: list,
        num_samples: int,
        sample_all: bool = False,
        prominent_amplitude: bool = False,
    ):
        super().__init__(files, num_samples, sample_all)
        self.prominent_amplitude = prominent_amplitude

    def sample_all_files(self) -> list:
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
                    if self.prominent_amplitude:
                        try:
                            prominent_pulses = block.data_arrays[
                                "prominent_pulses"
                            ]
                        except KeyError:
                            log.exception(
                                "No boolen Array found for max index please "
                                "save your config with enable max amplitude on"
                            )
                            sys.exit(1)
                        num_samples_per_file.append(np.sum(prominent_pulses))
                    else:
                        num_samples_per_file.append(data_array.shape[0])
            else:
                log.warning(
                    f"File {file} has unknown file type: {file.suffix}. Skipping."
                )
                # pbar.update(task, advance=1)
        return self._get_sample_indices(num_samples_per_file, self.files)

    def sample_single_file(self, filename: str) -> list:
        file = Path(
            [file for file in self.files if Path(file).name == filename][0]
        )
        if not file.exists:
            raise FileNotFoundError(f"{file.name} not found")

        num_samples_channels = []
        nixfile = nixio.File(str(file), nixio.FileMode.ReadOnly)

        if len(nixfile.blocks) == 0:
            log.warning(f"File {file} has no blocks.")
            exit(1)
        if len(nixfile.blocks) > 1:
            log.critical(
                f"File {file} has more than one block. Using the first one."
            )
        block = nixfile.blocks[0]
        data_array = block.data_arrays["pulses"]
        channels_unique = np.unique(block.data_arrays["channels"])
        channels = block.data_arrays["channels"]

        for ch in channels_unique:
            if self.prominent_amplitude:
                channel_index = channels == ch
                try:
                    prominent_pulses = block.data_arrays["prominent_pulses"]
                    prom_pulses = prominent_pulses[channel_index]
                    num_samples_channels.append(np.sum(prom_pulses))
                except KeyError:
                    log.exception(
                        """No boolen Array found for prominent index please
                        save your config with enable prominent amplitude on"""
                    )
                    sys.exit(1)
            else:
                num_samples_channels.append(np.sum(channels[:] == ch))

        return self._get_sample_indices(num_samples_channels, channels_unique)

    def _get_sample_indices(
        self, num_samples: Sequence[int], sampling_from: list
    ) -> list:
        log.info(
            f"Found {len(num_samples)} files with total of {intword(np.sum(num_samples))} peaks."
        )

        # Check if the total number of peaks fits the requested number of samples
        if np.sum(num_samples) < self.num_samples:
            raise ValueError(
                f"Not enough samples in the dataset. Found "
                f"{np.sum(num_samples)}, but requested "
                f"{self.num_samples}."
            )

        # Randomly sample peaks from each file to reach the requested number of samples
        total_frac = num_samples / np.sum(num_samples)
        target_samples = np.round(total_frac * self.num_samples).astype(int)

        sample_indices = []
        with get_progress() as pbar:
            task = pbar.add_task(
                "Sampling pulses from each channel", total=len(sampling_from)
            )
            for i in range(len(sampling_from)):
                num_samples_to_sample = target_samples[i]
                indices = np.sort(
                    np.random.choice(
                        num_samples[i], num_samples_to_sample, replace=False
                    )
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
            "--num-samples", "-n", help="Number of total samples to draw"
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
    singe_file: Annotated[
        bool,
        typer.Option(
            "--single-file",
            "-s",
            help="Sample only from a single file",
        ),
    ] = False,
    sample_all: Annotated[
        bool,
        typer.Option(
            "--sample-all",
            "-a",
            help="Take all samples",
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

    log.info(f"Found {len(data)} files in the dataset.")

    # sampler = StratifiedRandomSampler(data, num_samples=num_samples)
    sampler = StratifiedRandomSampler(data, num_samples, sample_all=sample_all)

    log.info(f"Sampling {num_samples} peaks from {len(data)} files.")

    if singe_file:
        log.info("Taking the first pulses.nix file")
        sample_indices = sampler.sample_single_file(data[0].name)
        sample_indices = [i.tolist() for i in sample_indices]
        data = [str(file.resolve()) for file in data]

    else:
        sample_indices = sampler.sample_all_files()
        sample_indices_index = [
            x for x in range(len(sample_indices)) if len(sample_indices[x]) > 0
        ]
        sample_indices = [
            sample_indices[i].tolist() for i in sample_indices_index
        ]
        data = [data[i] for i in sample_indices_index]
        data = [str(file.resolve()) for file in data]

    savepath = None
    if dtype == "file" or dtype in ["dir", "subdir"]:
        savepath = path.parent / (path.stem + "_samples")
    else:
        raise ValueError(f"Unknown dataset type: {dtype}")
    savepath = savepath.with_suffix(".json")

    # TODO: In Nix file

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
    outfile = savepath.with_suffix(".nix")

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

    if singe_file:
        file = data[0]
        for i in range(len(sample_indices)):
            indices = np.sort(sample_indices[i])
            log.info(f"Collecting {indices.size} samples from channel {i}.")
            f = nixio.File.open(str(file), nixio.FileMode.ReadOnly)
            if not check_nixfile(f, file):
                msg = f"File {file} is not a valid NIX file. Exiting."
                raise ValueError(msg)
            block = f.blocks[0]
            channels = block.data_arrays["channels"][:]
            ch_indices = np.sort(np.where(channels == i)[0])
            selected_indices = ch_indices[indices]
            for arr in block.data_arrays:
                name = arr.name
                if len(arr.shape) > 1:
                    arr = arr[:]
                vals = arr[selected_indices]
                data_array = sample_block.create_data_array(
                    f"{name}_{i}", "samples", data=vals
                )
    else:
        for i in range(len(sample_indices)):
            indices = np.sort(sample_indices[i])
            file = data[i]
            n = len(indices)
            files = [file] * n
            log.info(f"Collecting {n} samples from {file}.")
            f = nixio.File.open(str(file), nixio.FileMode.ReadOnly)
            if not check_nixfile(f, file):
                msg = f"File {file} is not a valid NIX file. Exiting."
                raise ValueError(msg)
            block = f.blocks[0]
            for arr in block.data_arrays:
                name = arr.name
                bindices = np.zeros(arr.shape[0], dtype=bool)
                bindices[indices] = True
                vals = arr[bindices]
                if i == 0:
                    data_array = sample_block.create_data_array(
                        name, "samples", data=vals
                    )
                else:
                    data_array = sample_block.data_arrays[name]
                    data_array.append(vals)

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
    pulses = dataset.blocks[0].data_arrays["pulses"][:]
    embedder = UmapEmbedder("umap", str(modelpath / "umap.joblib"))
    log.info("Fitting UMAP to the sampled data.")

    embedder.fit(pulses)
    log.info("Predicting UMAP embedding for the sampled data.")
    yhat = embedder.predict(pulses)

    plt.plot(yhat[:, 0], yhat[:, 1], "o", markersize=1)
    plt.show()

    block = dataset.blocks[0]
    data_array = block.create_data_array(
        "umap_embedding", "embedding", data=yhat
    )
    dataset.close()
    log.info("UMAP embedding saved to the file.")
