import dash_bootstrap_components as dbc
from dash import dcc, html


def create_preprocessing_offcanvas():
    preprocessing_card = dbc.Card(
        dbc.CardBody(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Pre filter parameters",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                dbc.Checklist(
                                    options=[
                                        {
                                            "label": "Apply Common Median Reference",
                                            "value": 0,
                                        }
                                    ],
                                    id="sw_common_reference",
                                    switch=True,
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
                            "Filter parameters",
                            id="h5_filter_operations",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
                                html.Div(
                                    dbc.Button(
                                        "Savitzky-Golay",
                                        color="info",
                                        id="bt_toggle_savgol_params",
                                        n_clicks=0,
                                    ),
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Window length [int]",
                                                    id="h6_savgol_window_length",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="int",
                                                    min=0,
                                                    id="num_savgol_window_length",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Polyorder [int]",
                                                    id="h6_savgol_polyorder",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="int",
                                                    min=0,
                                                    id="num_savgol_polyorder",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ]
                                        ),
                                        color="success",
                                        outline=True,
                                    ),
                                    id="collapse_savgol",
                                    is_open=False,
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Bandpass",
                                    color="info",
                                    id="bt_toggle_bandpass_params",
                                    n_clicks=0,
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Low cut off [int]",
                                                    id="h6_bandpass_lowcutoff",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="int",
                                                    min=1,
                                                    id="num_bandpass_lowcutoff",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "High cut off [int]",
                                                    id="h6_bandpass_highcutoff",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="int",
                                                    min=1,
                                                    id="num_bandpass_highcutoff",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ]
                                        ),
                                        color="success",
                                        outline=True,
                                    ),
                                    id="collapse_bandpass",
                                    is_open=False,
                                ),
                                html.Br(),
                                html.Br(),
                                dbc.Button(
                                    "Notch filter",
                                    color="info",
                                    id="bt_toggle_notchfilter_params",
                                    n_clicks=0,
                                ),
                                dbc.Collapse(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H6(
                                                    "Frequency",
                                                    id="h6_notchfilter_freq",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    min=1,
                                                    id="num_notchfilter_freq",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                                html.H6(
                                                    "Quality Factor",
                                                    id="h6_notchfilter_quality_factor",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    min=1,
                                                    id="num_notchfilter_quality",
                                                    style={"width": "70%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                ),
                                            ]
                                        ),
                                        color="success",
                                        outline=True,
                                    ),
                                    id="collapse_notchfilter",
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
                            "Pulse detection parameters ",
                            className="card-title",
                            id="h5_pulse_dectection_parameters",
                            style={"textAlign": "center"},
                        ),
                        dbc.CardBody(
                            [
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
                                                ),
                                                html.H6(
                                                    "Time for grouping pluses across channels [seconds]",
                                                    id="h6_pluse_general",
                                                ),
                                                dbc.Input(
                                                    type="number",
                                                    placeholder="float",
                                                    min=0,
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
                                                    min=0,
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
                                                    min=1,
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
                                                    min=1,
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
                                                    min=0,
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
                                                    min=0,
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
                                                    min=0,
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
                            "Resample parameters",
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
                                                    min=0,
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
                # dbc.Card(
                #     dbc.CardBody(
                #         [
                #             html.H5(
                #                 "General parameters",
                #                 className="card-title",
                #                 id="h5_general_parameters",
                #                 style={"textAlign": "center"},
                #             ),
                #             html.H6(
                #                 "Buffersize for loading in the recording",
                #                 id="h6_general_buffersize",
                #             ),
                #             dbc.Input(
                #                 type="number",
                #                 placeholder="int",
                #                 min=0,
                #                 id="num_general_buffersize",
                #                 style={"width": "70%"},
                #                 persistence=True,
                #                 persistence_type="local",
                #             ),
                #         ]
                #     )
                # ),
                # dbc.Checklist(
                #     options=[
                #         {"label": "Search peak in Current window", "value": 0}
                #     ],
                #     id="sw_peaks_current_window",
                #     switch=True,
                # ),
                # html.Br(),
                # html.P(
                #     children="Muliply the threshold by N",
                #     style={"textAlign": "center"},
                #     id="p_threshold",
                # ),
                # dcc.Slider(
                #     1,
                #     10,
                #     0.1,
                #     marks=None,
                #     id="n_median",
                #     updatemode="drag",
                #     value=4,
                #     tooltip={"placement": "bottom"},
                #     persistence=True,
                #     persistence_type="local",
                # ),
                # html.Br(),
                # html.P(
                #     children="The minimal distance from channel to channel is ",
                #     id="minimum_distance_channel",
                #     style={"textAlign": "center"},
                # ),
                # html.P(
                #     children="Exclude Radius for peak detection in [um]",
                #     style={"textAlign": "center"},
                # ),
                # dbc.Input(
                #     type="number",
                #     placeholder="Exclude Radius",
                #     min=3,
                #     id="exclude_radius",
                #     style={"width": "35%"},
                #     persistence=True,
                #     persistence_type="local",
                # ),
                # html.Br(),
                # dbc.Checklist(
                #     options=[{"label": "Show merged peaks", "value": 0}],
                #     id="sw_merged_peaks",
                #     switch=True,
                # ),
                # html.Br(),
                # html.P(
                #     children="Remove Artefacts from data",
                #     style={"textAlign": "center"},
                # ),
                # dbc.Input(
                #     type="number",
                #     placeholder="Thresh Artefacts",
                #     id="threshold_artefact",
                #     style={"width": "35%"},
                #     persistence=True,
                #     persistence_type="local",
                # ),
                # html.Br(),
                # html.P(
                #     children="Chunksize for processessing in seconds",
                #     style={"textAlign": "center"},
                # ),
                # dbc.Input(
                #     type="number",
                #     placeholder="Chucksize",
                #     min=1,
                #     id="chunksize",
                #     style={"width": "35%"},
                #     persistence=True,
                #     persistence_type="local",
                #     disabled=True,
                # ),
                # html.Br(),
                # dbc.Button(
                #     "Save Preproccessing",
                #     color="info",
                #     id="bt_save_to_disk",
                #     n_clicks=0,
                # ),
                # html.Br(),
                # dbc.Spinner(
                #     html.Div(id="spinner_save_preprocessing"), color="info"
                # ),
                # html.Br(),
                # html.P(
                #     children="You switch to the saved processed data with this switch",
                # ),
                # dbc.Checklist(
                #     options=[{"label": "Show processed data", "value": 0}],
                #     id="sw_processed",
                #     switch=True,
                # ),
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
            className="mx-auto",
        )
    )
    return preprocessing_card
