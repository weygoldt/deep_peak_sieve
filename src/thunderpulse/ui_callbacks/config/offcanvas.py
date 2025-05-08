from dash import Dash, Input, Output, State


def callbacks(app: Dash) -> None:
    @app.callback(
        Output("io_offcanvas", "is_open"),
        Input("bt_io_offcanvas", "n_clicks"),
        State("io_offcanvas", "is_open"),
    )
    def toggle_io_offcanvas(n_clicks: int | None, is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("filter_offcanvas", "is_open"),
        Input("bt_filter_offcanvas", "n_clicks"),
        State("filter_offcanvas", "is_open"),
    )
    def toggle_filter_offcanvas(n_clicks: int | None, is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("pulse_offcanvas", "is_open"),
        Input("bt_pulse_offcanvas", "n_clicks"),
        State("pulse_offcanvas", "is_open"),
    )
    def toggle_pulse_offcanvas(n_clicks: int | None, is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output("classificaion_offcanvas", "is_open"),
        Input("bt_classification_offcanvas", "n_clicks"),
        State("classificaion_offcanvas", "is_open"),
    )
    def toggle_pulse_offcanvas(n_clicks: int | None, is_open: bool) -> bool:
        if n_clicks:
            return not is_open
        return is_open
