import dash_bootstrap_components as dbc
from dash import html

from .cluster_card import create_cluster_card
from .io_card import create_io_offcanvas
from .peak_table_card import create_peak_table_card
from .preprocessing_card import create_preprocessing_offcanvas


def create_config_navbar():
    io_offcanvas = create_io_offcanvas()
    preprocessing_offcanvas = create_preprocessing_offcanvas()
    # peak_table_card = create_peak_table_card()
    # cluster_card = create_cluster_card()

    navbar = dbc.Container(
        [
            dbc.Row(dbc.Col(html.A("Configurations")), justify="center"),
            dbc.Row(
                dbc.Navbar(
                    dbc.Container(
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
                                                "Preprocessing",
                                                color="info",
                                                id="bt_preprocessing_offcanvas",
                                                n_clicks=0,
                                            ),
                                            dbc.Offcanvas(
                                                preprocessing_offcanvas,
                                                title="Pulse detection preprocessing and detection parameter",
                                                is_open=False,
                                                id="preprocessing_offcanvas",
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
                                ],
                                align="center",
                                className="g-0",
                            ),
                        ],
                    ),
                    color="dark",
                    dark=True,
                ),
                align="center",
            ),
        ]
    )

    return navbar
