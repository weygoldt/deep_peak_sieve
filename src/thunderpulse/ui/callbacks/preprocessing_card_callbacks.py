import nixio
import numpy as np
from dash import Dash, Input, Output


def callbacks(app: Dash):
    @app.callback(
        Output("minimum_distance_channel", "children"),
        Input("filepath", "data"),
    )
    def minimum_distance_channel(filepath):
        if not filepath:
            return None
        if not filepath["data_path"]:
            return None
        nix_file = nixio.File.open(
            filepath["data_path"], nixio.FileMode.ReadOnly
        )
        probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        sorted_y = np.argsort(probe_frame["y"])
        x = probe_frame["x"][sorted_y]
        y = probe_frame["y"][sorted_y]
        points = np.vstack((x, y)).T
        differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=2)
        minimum = np.sort(distances, axis=0)[1]
        nix_file.close()

        return (
            f"The minimal distance from channel to channel is {minimum[0]} um"
        )

    @app.callback(
        Output("p_threshold", "children"), Input("n_median", "value")
    )
    def update_threshold(value):
        return f"muliply the threshold by: {value}"
