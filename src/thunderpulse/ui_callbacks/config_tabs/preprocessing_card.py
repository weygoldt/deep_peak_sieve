import nixio
import numpy as np
from dash import Dash, Input, Output, State

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

    @app.callback(
        Output("collapse_savgol", "is_open"),
        [Input("bt_toggle_savgol_params", "n_clicks")],
        [State("collapse_savgol", "is_open")],
    )
    def toggle_collapse_savgol(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse_bandpass", "is_open"),
        [Input("bt_toggle_bandpass_params", "n_clicks")],
        [State("collapse_bandpass", "is_open")],
    )
    def toggle_collapse_bandpass(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse_notchfilter", "is_open"),
        [Input("bt_toggle_notchfilter_params", "n_clicks")],
        [State("collapse_notchfilter", "is_open")],
    )
    def toggle_collapse_notchfilter(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse_pulse", "is_open"),
        [Input("bt_toggle_pulse_params", "n_clicks")],
        [State("collapse_pulse", "is_open")],
    )
    def toggle_collapse_pulse(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse_findpeaks", "is_open"),
        [Input("bt_toggle_findpeaks_params", "n_clicks")],
        [State("collapse_findpeaks", "is_open")],
    )
    def toggle_collapse_findpeaks(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        Output("collapse_resample", "is_open"),
        [Input("bt_toggle_resample_params", "n_clicks")],
        [State("collapse_resample", "is_open")],
    )
    def toggle_collapse_resample(n, is_open):
        if n:
            return not is_open
        return is_open
