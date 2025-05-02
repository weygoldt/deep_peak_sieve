import numpy as np
from dash import Dash, Input, Output, State

from thunderpulse.data_handling.data import load_data


def callbacks(app: Dash) -> None:
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

