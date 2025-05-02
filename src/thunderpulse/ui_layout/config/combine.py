import dash_bootstrap_components as dbc
from dash import html

from thunderpulse.ui_layout.config.filter_card import (
    create_filter_offcanvas,
)
from thunderpulse.ui_layout.config.paths_card import create_paths_offcanvas
from thunderpulse.ui_layout.config.pulse_detection import (
    create_pulse_offcanvas,
)


def create_config():
    io_offcanvas = create_paths_offcanvas()
    filter_offcanvas = create_filter_offcanvas()
    pulse_offcanvas = create_pulse_offcanvas()

    navbar = dbc.Card(
        [
            dbc.CardHeader("Configurations", style={"textAlign": "center"}),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Paths",
                                        color="info",
                                        id="bt_io_offcanvas",
                                        n_clicks=0,
                                    ),
                                    dbc.Offcanvas(
                                        io_offcanvas,
                                        title="IO operations",
                                        is_open=False,
                                        id="io_offcanvas",
                                        backdrop=False,
                                    ),
                                ],
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Filter",
                                        color="info",
                                        id="bt_filter_offcanvas",
                                        n_clicks=0,
                                    ),
                                    dbc.Offcanvas(
                                        filter_offcanvas,
                                        title="Filter Parameters",
                                        is_open=False,
                                        id="filter_offcanvas",
                                        backdrop=False,
                                        scrollable=True,
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Detection",
                                        color="info",
                                        id="bt_pulse_offcanvas",
                                        n_clicks=0,
                                    ),
                                    dbc.Offcanvas(
                                        pulse_offcanvas,
                                        title="Pulse detection",
                                        is_open=False,
                                        id="pulse_offcanvas",
                                        backdrop=False,
                                        scrollable=True,
                                    ),
                                ]
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Load",
                                    color="warning",
                                    id="bt_load_data",
                                    n_clicks=0,
                                ),
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Save",
                                    color="warning",
                                    id="bt_save_config",
                                    n_clicks=0,
                                ),
                            ),
                        ],
                        class_name="mx-auto",
                    ),
                ]
            ),
        ],
        color="dark",
        inverse=True,
    )

    return navbar
