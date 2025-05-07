import matplotlib.pyplot as plt
from typing import Annotated
from pathlib import Path
from thunderpulse.utils.loggers import get_logger, configure_logging
from thunderpulse.nn.embedders import UmapEmbedder
import numpy as np
import nixio
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)
log = get_logger(__name__)


@app.command()
def main(
    datapath: Annotated[Path, typer.Argument(help="Path to the data file")],
    verbose: Annotated[
        int, typer.Option(help="Verbosity level", count=True)
    ] = 0,
) -> None:
    configure_logging(verbose)

    dataset = nixio.File.open(
        str(datapath),
        nixio.FileMode.ReadOnly,
    )

    modelpath = datapath.parent

    pulses = dataset.blocks[0].data_arrays["pulses"]
    print(pulses.shape)

    embedder = UmapEmbedder("umap", str(modelpath / "umap.joblib"))
    embedder.fit(pulses)

    # TODO: Why are there only nans in my data? If nan, this should not be saved at all


if __name__ == "__main__":
    app()
