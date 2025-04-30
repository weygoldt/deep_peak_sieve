import nixio
import numpy as np
from dash import Dash, Input, Output

from thunderpulse.data_handling.data import load_data


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
        d = load_data(**filepath)
        sorted_y = np.argsort(d.sensorarray.y)
        x = d.sensorarray.y[sorted_y]
        y = d.sensorarray.x[sorted_y]
        points = np.vstack((x, y)).T
        differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.linalg.norm(differences, axis=2)
        minimum = np.sort(distances, axis=0)[1]

        return (
            f"The minimal distance from channel to channel is {minimum[0]} um"
        )

    @app.callback(
        Output("p_threshold", "children"), Input("n_median", "value")
    )
    def update_threshold(value):
        return f"muliply the threshold by: {value}"
