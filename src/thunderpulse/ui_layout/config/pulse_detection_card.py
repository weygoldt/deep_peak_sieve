import dash_bootstrap_components as dbc
from dash import dcc, html


def create_pulse_offcanvas() -> dbc.Card:
    """Generarte the preprocessing card."""
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Pulse detection parameters ",
                            className="card-title",
                            id="h5_pulse_dectection_parameters",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                dbc.Checklist(
                                    options=[
                                        {
                                            "label": "Detect pulses in current window",
                                            "value": 1,
                                        }
                                    ],
                                    id="sw_detect_pulses",
                                    switch=True,
                                    persistence=True,
                                    persistence_type="local",
                                ),
                                html.Br(),
                                dbc.Button(
                                    "General pulse detection",
                                    color="info",
                                    id="bt_toggle_pulse_params",
                                    n_clicks=0,
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Buffersize for detection [s]",
                                                    id="h6_pluse_min_channels",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    min=1,
                                                    id="num_pulse_buffersize",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Number of channels where pulses should be detected",
                                                    id="h6_pluse_min_channels",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="int",
                                                    min=1,
                                                    id="num_pulse_min_channels",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Mode of pulse detection,[peak, trough, both]",
                                                    id="h6_pluse_mode",
                                                ),
                                                dbc.Select(
                                                    id="select_pulse_mode",
                                                    options=[
                                                        {
                                                            "label": "Both",
                                                            "value": "both",
                                                        },
                                                        {
                                                            "label": "Peak",
                                                            "value": "peak",
                                                        },
                                                        {
                                                            "label": "Trough",
                                                            "value": "trough",
                                                        },
                                                    ],
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Time for grouping pluses across channels [seconds]",
                                                    id="h6_pluse_general",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_pulse_min_peak_distance",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Waveform duration from the center of the pulse [seconds]",
                                                    id="h6_pluse_waveform",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_pulse_waveform",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ]
                                        ),
                                        color="success",
                                        outline=True,
                                    ),
                                    id="collapse_pulse",
                                    is_open=False,
                                ),
                                html.Br(),
                                html.Br(),
                                dbc.Button(
                                    "Find peaks [scipy]",
                                    color="info",
                                    id="bt_toggle_findpeaks_params",
                                    n_clicks=0,
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Height of the Pulse",
                                                    id="h6_findpeaks_height",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_findpeaks_height",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Threshold",
                                                    id="h6_findpeaks_threshold",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_findpeaks_threshold",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Distance",
                                                    id="h6_findpeaks_distance",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_findpeaks_distance",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Prominence",
                                                    id="h6_findpeaks_prominence",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_findpeaks_prominence",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Width",
                                                    id="h6_findpeaks_width",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    id="num_findpeaks_width",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ]
                                        ),
                                        color="success",
                                        outline=True,
                                    ),
                                    id="collapse_findpeaks",
                                    is_open=False,
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
                            "Note",
                            className="card-title",
                            id="h5_pulse_detection_parameters",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                html.P(
                                    "After pulse detection, a mean waveform of the pulse is calculated using as many channels as the pulse simultaneously occured on. The following parameters govern how this mean waveform is calculated.",
                                    id="h6_pulse_detection_note",
                                )
                            ]
                        ),
                    ]
                ),
                html.Br(),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Mean pulse post-processing parameters",
                            className="card-title",
                            id="h5_resample_parameters",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                dbc.Button(
                                    "Resample",
                                    color="info",
                                    id="bt_toggle_resample_params",
                                    n_clicks=0,
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dbc.Checklist(
                                                    options=[
                                                        {
                                                            "label": "Enable resampling of waveforms",
                                                            "value": 0,
                                                        }
                                                    ],
                                                    id="sw_resampling_enable",
                                                    switch=True,
                                                ),
                                                html.H6(
                                                    "Numbers to which the waveforms are resampled",
                                                    id="h6_resample_n",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="int",
                                                    id="num_resampling_n",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ]
                                        ),
                                        color="success",
                                        outline=True,
                                    ),
                                    id="collapse_resample",
                                    is_open=False,
                                ),
                            ]
                        ),
                    ],
                    color="primary",
                    outline=True,
                ),
            ],
        )
    )
