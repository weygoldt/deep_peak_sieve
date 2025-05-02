import dash_bootstrap_components as dbc
from dash import html


def create_cluster_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    children="Waveform lower and higher limit",
                    style={"textAlign": "center"},
                ),
                dbc.Input(
                    type="number",
                    placeholder="lower ex. (-2ms)",
                    max=0,
                    id="waveform_lower",
                    style={"width": "50%"},
                    persistence=True,
                    persistence_type="local",
                ),
                dbc.Input(
                    type="number",
                    placeholder="higher ex. (2 ms)",
                    min=0,
                    id="waveform_higher",
                    style={"width": "50%"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                dbc.Button(
                    "Create Waveform Array",
                    color="info",
                    id="bt_save_waveforms",
                    n_clicks=0,
                ),
                dbc.Spinner(
                    html.Div(id="spinner_waveforms"),
                    color="info",
                ),
                html.Br(),
                html.P(
                    children="Number random Waveforms, per channel",
                    style={"textAlign": "center"},
                ),
                dbc.Input(
                    type="number",
                    placeholder="num Waveforms",
                    min=0,
                    id="waveform_rand_num",
                    style={"width": "50%"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                html.P(
                    children="Input channel for the umap",
                    style={"textAlign": "center"},
                ),
                dbc.Button(
                    "Calculate Umap embedding",
                    color="info",
                    id="bt_umap_embedding",
                    n_clicks=0,
                ),
                html.Br(),
                dbc.Spinner(html.Div(id="spinner_umap"), color="info"),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    children="Input channel for the umap",
                                    style={"textAlign": "center"},
                                ),
                                dbc.Input(
                                    type="number",
                                    placeholder="Channel",
                                    min=0,
                                    max=32,
                                    id="num_channel_umap",
                                    # style={"width": "50%"},
                                    persistence=True,
                                    persistence_type="local",
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                html.P(
                                    children="HDB Scan Cluster size",
                                    style={"textAlign": "center"},
                                ),
                                dbc.Input(
                                    type="number",
                                    placeholder="hdb_scan",
                                    min=5,
                                    # max=32,
                                    id="num_hdb_scan",
                                    # style={"width": "50%"},
                                    persistence=True,
                                    persistence_type="local",
                                ),
                            ]
                        ),
                    ]
                ),
                html.Br(),
                dbc.Button(
                    "Show Umap Channel",
                    color="info",
                    id="bt_show_umap_channel",
                    n_clicks=0,
                ),
                html.Br(),
                dbc.Checklist(
                    options=[{"label": "Show Segments", "value": 0}],
                    id="sw_segments",
                    switch=True,
                ),
                html.Br(),
                dbc.Button(
                    "Save Unit",
                    color="info",
                    id="bt_saveunit",
                    n_clicks=0,
                ),
            ]
        ),
    )
