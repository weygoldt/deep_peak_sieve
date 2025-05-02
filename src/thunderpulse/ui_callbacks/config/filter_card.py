import numpy as np
from dash import Dash, Input, Output, State

from thunderpulse.data_handling.data import load_data


def callbacks(app: Dash) -> None:
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

