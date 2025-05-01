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
        Output("preprocessing_offcanvas", "is_open"),
        Input("bt_preprocessing_offcanvas", "n_clicks"),
        State("preprocessing_offcanvas", "is_open"),
    )
    def toggle_preprocessing_offcanvas(
        n_clicks: int | None, is_open: bool
    ) -> bool:
        if n_clicks:
            return not is_open
        return is_open
