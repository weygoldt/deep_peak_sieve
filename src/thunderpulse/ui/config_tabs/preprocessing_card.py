import dash_bootstrap_components as dbc
from dash import dcc, html


def create_preprocessing_card():
    preprocessing_card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Checklist(
                    options=[
                        {"label": "Apply Common Median Reference", "value": 0}
                    ],
                    id="sw_common_reference",
                    switch=True,
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                dbc.Checklist(
                    options=[{"label": "Apply Bandbass Filter", "value": 0}],
                    id="sw_bandpass_filter",
                    switch=True,
                    persistence=True,
                    persistence_type="local",
                ),
                dbc.Input(
                    type="number",
                    placeholder="LowCutOff [int]",
                    min=1,
                    id="lowcutoff",
                    style={"width": "35%"},
                    persistence=True,
                    persistence_type="local",
                ),
                dbc.Input(
                    type="number",
                    placeholder="HighCutOff [int]",
                    min=1,
                    id="highcutoff",
                    style={"width": "35%"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                dbc.Checklist(
                    options=[{"label": "Apply Notch Filter", "value": 0}],
                    id="sw_notch_filter",
                    switch=True,
                    persistence=True,
                    persistence_type="local",
                ),
                dbc.Input(
                    type="number",
                    placeholder="NotchFilter",
                    min=1,
                    id="notch",
                    style={"width": "35%"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                html.Br(),
                dbc.Checklist(
                    options=[
                        {"label": "Search peak in Current window", "value": 0}
                    ],
                    id="sw_peaks_current_window",
                    switch=True,
                ),
                html.Br(),
                html.P(
                    children="Muliply the threshold by N",
                    style={"textAlign": "center"},
                    id="p_threshold",
                ),
                dcc.Slider(
                    1,
                    10,
                    0.1,
                    marks=None,
                    id="n_median",
                    updatemode="drag",
                    value=4,
                    tooltip={"placement": "bottom"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                html.P(
                    children="The minimal distance from channel to channel is ",
                    id="minimum_distance_channel",
                    style={"textAlign": "center"},
                ),
                html.P(
                    children="Exclude Radius for peak detection in [um]",
                    style={"textAlign": "center"},
                ),
                dbc.Input(
                    type="number",
                    placeholder="Exclude Radius",
                    min=3,
                    id="exclude_radius",
                    style={"width": "35%"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                dbc.Checklist(
                    options=[{"label": "Show merged peaks", "value": 0}],
                    id="sw_merged_peaks",
                    switch=True,
                ),
                html.Br(),
                html.P(
                    children="Remove Artefacts from data",
                    style={"textAlign": "center"},
                ),
                dbc.Input(
                    type="number",
                    placeholder="Thresh Artefacts",
                    id="threshold_artefact",
                    style={"width": "35%"},
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
                html.P(
                    children="Chunksize for processessing in seconds",
                    style={"textAlign": "center"},
                ),
                dbc.Input(
                    type="number",
                    placeholder="Chucksize",
                    min=1,
                    id="chunksize",
                    style={"width": "35%"},
                    persistence=True,
                    persistence_type="local",
                    disabled=True,
                ),
                html.Br(),
                dbc.Button(
                    "Save Preproccessing",
                    color="info",
                    id="bt_save_to_disk",
                    n_clicks=0,
                ),
                html.Br(),
                dbc.Spinner(
                    html.Div(id="spinner_save_preprocessing"), color="info"
                ),
                html.Br(),
                html.P(
                    children="You switch to the saved processed data with this switch",
                ),
                dbc.Checklist(
                    options=[{"label": "Show processed data", "value": 0}],
                    id="sw_processed",
                    switch=True,
                ),
                # html.Br(),
                # dbc.Input(
                #     type="number",
                #     placeholder="Spike ampl winsize",
                #     min=10,
                #     id="num_windowsize_ampl",
                #     style={"width": "35%"},
                #     persistence=True,
                #     persistence_type="local",
                # ),
                # dbc.Button(
                #     "Save Mean Spike Ampl",
                #     color="info",
                #     id="bt_save_mean_spike_ampl",
                #     n_clicks=0,
                # ),
                # html.Br(),
            ],
        )
    )
    return preprocessing_card
