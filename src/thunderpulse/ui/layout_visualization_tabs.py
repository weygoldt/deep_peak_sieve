import dash_bootstrap_components as dbc
from dash import dcc


def create_visualization_tabs():
    tabs = dbc.Tabs(
        [
            dbc.Tab(
                dcc.Graph(
                    id="traces",
                    responsive=True,
                    style={"height": "75vh"},
                    config={"frameMargins": 0.0},
                ),
                label="Traces",
                tab_id="tab_traces",
            ),
            dbc.Tab(
                dcc.Graph(
                    id="psd",
                    responsive=True,
                    style={"height": "75vh"},
                    config={"frameMargins": 0.0},
                ),
                label="Spectral Density",
                tab_id="tab_psd",
            ),
            dbc.Tab(
                dcc.Graph(
                    id="waveforms",
                    responsive=True,
                    style={"height": "75vh"},
                    config={"frameMargins": 0.0},
                ),
                label="Waveforms",
                tab_id="tab_waveforms",
            ),
            dbc.Tab(
                dcc.Graph(
                    id="spikes_ampl",
                    responsive=True,
                    style={"height": "75vh"},
                    config={"frameMargins": 0.0},
                ),
                label="Spike-Amplitudes",
                tab_id="tab_spikes_ampl",
            ),
            dbc.Tab(
                [
                    dcc.Graph(
                        id="umap",
                        responsive=True,
                        style={"height": "30vh"},
                        config={"frameMargins": 0.0},
                    ),
                    dcc.Graph(
                        id="umap_selection",
                        responsive=True,
                        style={"height": "30vh"},
                        config={"frameMargins": 0.0},
                    ),
                ],
                label="Umap",
                tab_id="tab_umap",
            ),
        ],
        id="vis_tabs",
    )
    return tabs
