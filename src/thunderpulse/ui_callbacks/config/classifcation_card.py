from pathlib import Path

import nixio
import numpy as np
from dash import Dash, Input, Output, State

from thunderpulse.data_handling.data import load_data


def callbacks(app: Dash) -> None:
    @app.callback(
        Output("select_umap_embedding", "options"),
        Input("filepath", "data"),
    )
    def add_umap_embeddings(filepath):
        if not filepath:
            return None
        if not filepath["data_path"]:
            return None

        d = load_data(**filepath)

        save_path = list(Path(d.paths.save_path).rglob("*pulses.*"))
        try:
            save_file = [p for p in save_path if p.suffix in [".nix", ".h5"]][
                0
            ]
        except IndexError:
            return None

        if not save_file.exists:
            return None
        nix_file = nixio.File(str(save_file), nixio.FileMode.ReadOnly)
        block = nix_file.blocks[0]
        embdeddings = [
            {"label": arr.name, "value": arr.name}
            for arr in block.data_arrays
            if "umap_embedding" in arr.name
        ]
        return embdeddings
