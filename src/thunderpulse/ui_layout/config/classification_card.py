import dash_bootstrap_components as dbc
from dash import dcc, html


def create_classification_offcanvas() -> dbc.Card:
    """Generarte the classificaiton card."""
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Pulse sampler",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                html.H6(
                                    "Number of samples",
                                    id="h6_sampler",
                                ),
                                dbc.Input(
                                    type="number",
                                    placeholder="float",
                                    min=0,
                                    id="num_pulse_sampler",
                                    style={"width": "70%"},
                                    persistence=True,
                                    persistence_type="local",
                                ),
                            ]
                        ),
                    ],
                    color="primary",
                    outline=True,
                ),
                html.Br(),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Umap Selection",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                html.H6(
                                    "Which umap embedding",
                                ),
                                dbc.Select(
                                    id="select_umap_embedding",
                                    options=[],
                                    persistence=True,
                                    persistence_type="local",
                                ),
                                # html.H6(
                                #     "Unit Name",
                                # ),
                                # dbc.Input(
                                #     type="number",
                                #     placeholder="float",
                                #     min=0,
                                #     id="num_umap_unit_name",
                                #     style={"width": "70%"},
                                #     persistence=True,
                                #     persistence_type="local",
                                # ),
                            ]
                        ),
                    ],
                    color="primary",
                    outline=True,
                ),
                html.Br(),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "HDBSCAN",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                html.H6(
                                    "Minimum Cluster Size",
                                ),
                                dbc.Input(
                                    type="number",
                                    placeholder="float",
                                    min=0,
                                    id="num_hdbscan_cluster_size",
                                    style={"width": "70%"},
                                    persistence=True,
                                    persistence_type="local",
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Run HDBSCAN",
                                    color="info",
                                    id="bt_run_hdbscan",
                                    n_clicks=0,
                                ),
                            ]
                        ),
                    ],
                    color="primary",
                    outline=True,
                ),
                html.Br(),
                dbc.Button(
                    "Save Unit",
                    color="warning",
                    id="bt_umap_save_unit",
                    n_clicks=0,
                ),
            ],
        )
    )
