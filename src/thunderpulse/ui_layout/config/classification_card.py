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
                                html.H6(
                                    "Unit Name",
                                ),
                                dbc.Input(
                                    type="number",
                                    placeholder="float",
                                    min=0,
                                    id="num_umap_unit_name",
                                    style={"width": "70%"},
                                    persistence=True,
                                    persistence_type="local",
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Save Unit",
                                    color="info",
                                    id="bt_umap_save_unit",
                                    n_clicks=0,
                                ),
                            ]
                        ),
                    ],
                    color="primary",
                    outline=True,
                ),
                html.Br(),
            ],
        )
    )
    # dbc.Card(
    #     [
    #         dbc.CardHeader(
    #             "Filter parameters",
    #             id="h5_filter_operations",
    #             style={"textAlign": "center"},
    #         ),
    #         dbc.CardBody(
    #             [
    #                 html.Div(
    #                     dbc.Button(
    #                         "Savitzky-Golay",
    #                         color="info",
    #                         id="bt_toggle_savgol_params",
    #                         n_clicks=0,
    #                     ),
    #                 ),
    #                 dbc.Collapse(
    #                     dbc.Card(
    #                         dbc.CardBody(
    #                             [
    #                                 html.H6(
    #                                     "Window length [float] in seconds",
    #                                     id="h6_savgol_window_length",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="float",
    #                                     min=0,
    #                                     id="num_savgol_window_length",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                                 html.H6(
    #                                     "Polyorder [int]",
    #                                     id="h6_savgol_polyorder",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="int",
    #                                     min=0,
    #                                     id="num_savgol_polyorder",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                             ]
    #                         ),
    #                         color="success",
    #                         outline=True,
    #                     ),
    #                     id="collapse_savgol",
    #                     is_open=False,
    #                 ),
    #                 html.Br(),
    #                 dbc.Button(
    #                     "Bandpass",
    #                     color="info",
    #                     id="bt_toggle_bandpass_params",
    #                     n_clicks=0,
    #                 ),
    #                 dbc.Collapse(
    #                     dbc.Card(
    #                         dbc.CardBody(
    #                             [
    #                                 html.H6(
    #                                     "Low cut off [int]",
    #                                     id="h6_bandpass_lowcutoff",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="int",
    #                                     min=1,
    #                                     id="num_bandpass_lowcutoff",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                                 html.H6(
    #                                     "High cut off [int]",
    #                                     id="h6_bandpass_highcutoff",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="int",
    #                                     min=1,
    #                                     id="num_bandpass_highcutoff",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                                 html.H6(
    #                                     "Order of the filter",
    #                                     id="h6_bandpass_order",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="int",
    #                                     min=1,
    #                                     id="num_bandpass_order",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                             ]
    #                         ),
    #                         color="success",
    #                         outline=True,
    #                     ),
    #                     id="collapse_bandpass",
    #                     is_open=False,
    #                 ),
    #                 html.Br(),
    #                 html.Br(),
    #                 dbc.Button(
    #                     "Notch filter",
    #                     color="info",
    #                     id="bt_toggle_notchfilter_params",
    #                     n_clicks=0,
    #                 ),
    #                 dbc.Collapse(
    #                     dbc.Card(
    #                         dbc.CardBody(
    #                             [
    #                                 html.H6(
    #                                     "Frequency",
    #                                     id="h6_notchfilter_freq",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="float",
    #                                     min=1,
    #                                     id="num_notchfilter_freq",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                                 html.H6(
    #                                     "Quality Factor",
    #                                     id="h6_notchfilter_quality_factor",
    #                                 ),
    #                                 dbc.Input(
    #                                     type="number",
    #                                     placeholder="float",
    #                                     min=1,
    #                                     id="num_notchfilter_quality",
    #                                     style={"width": "70%"},
    #                                     persistence=True,
    #                                     persistence_type="local",
    #                                 ),
    #                             ]
    #                         ),
    #                         color="success",
    #                         outline=True,
    #                     ),
    #                     id="collapse_notchfilter",
    #                     is_open=False,
    #                 ),
    #             ]
    #         ),
