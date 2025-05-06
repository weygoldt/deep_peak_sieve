import pathlib
import re

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from IPython import embed
from IPython.core.interactiveshell import is_integer_string
from plotly import subplots

from thunderpulse.data_handling.data import load_data
from thunderpulse.pulse_detection.config import (
    BandpassParameters,
    FiltersParameters,
    FindPeaksKwargs,
    NotchParameters,
    Params,
    PeakDetectionParameters,
    PrefilterParameters,
    ResampleParameters,
    SavgolParameters,
)
from thunderpulse.pulse_detection.detection import (
    apply_filters,
    detect_peaks_on_block,
)
from thunderpulse.ui_callbacks.graphs.channel_selection import select_channels
from thunderpulse.utils.check_config import check_config_params

# from thunderpulse.utils.cleaning import remove_none_inputs
from thunderpulse.utils.loggers import get_logger

from . import data_selection as ds

log = get_logger(__name__)


def default_traces_figure():
    fig = subplots.make_subplots(
        rows=16,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    fig.update_layout(
        showlegend=False,
        clickmode="event+select",
        autosize=True,
        template="plotly_dark",
    )
    return fig


def callbacks_traces(app):
    @app.callback(
        [Output("traces", "figure"), Output("peak_storage", "data")],
        # Filter
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
                "vis_tabs": Input("vis_tabs", "active_tab"),
                "time_slider": Input("time_slider", "value"),
                "channels": Input("channel_range_slider", "value"),
                "probe_selected_channels": Input("probe", "selectedData"),
                "detect_pulses": Input("sw_detect_pulses", "value"),
            },
            "pre_filter": {
                "common_median_reference": Input(
                    "sw_common_reference", "value"
                ),
            },
            "savgol": {
                "window_length_s": Input("num_savgol_window_length", "value"),
                "polyorder": Input("num_savgol_polyorder", "value"),
            },
            "bandpass": {
                "lowcut": Input("num_bandpass_lowcutoff", "value"),
                "highcut": Input("num_bandpass_highcutoff", "value"),
            },
            "notch": {
                "notch_freq": Input("num_notchfilter_freq", "value"),
                "quality_factor": Input("num_notchfilter_quality", "value"),
            },
            "pulse": {
                "min_channels": Input("num_pulse_min_channels", "value"),
                "mode": Input("select_pulse_mode", "value"),
                "min_peak_distance_s": Input(
                    "num_pulse_min_peak_distance", "value"
                ),
                "cutout_window_around_peak_s": Input(
                    "num_pulse_waveform", "value"
                ),
            },
            "findpeaks": {
                "height": Input("num_findpeaks_height", "value"),
                "threshold": Input("num_findpeaks_threshold", "value"),
                "distance": Input("num_findpeaks_distance", "value"),
                "prominence": Input("num_findpeaks_prominence", "value"),
                "width": Input("num_findpeaks_width", "value"),
            },
            "resample": {
                "enabled": Input("sw_resampling_enable", "value"),
                "n_resamples": Input("num_resampling_n", "value"),
            },
        },
    )
    def update_graph_traces(
        general,
        pre_filter,
        savgol,
        bandpass,
        notch,
        pulse,
        findpeaks,
        resample,
    ):
        (
            filepath,
            tabs,
            time_index,
            channels,
            probe_selected_channels,
            detect_pulses,
        ) = general.values()
        if tabs and tabs != "tab_traces":
            return default_traces_figure(), None
        if not filepath:
            return default_traces_figure(), None
        if not filepath["data_path"]:
            return default_traces_figure(), None

        d = load_data(**filepath)

        # NOTE: rewrite checkbox input [1]/True []/False to bool
        pre_filter["common_median_reference"] = bool(
            pre_filter["common_median_reference"]
        )
        prefilter = PrefilterParameters(**pre_filter)

        apply_filters_names = []
        apply_filters_params = []
        filter_params_function = [savgol, bandpass, notch]
        filter_names = FiltersParameters().filters
        filter_params = [SavgolParameters, BandpassParameters, NotchParameters]
        for f_name, f_params, f_params_func in zip(
            filter_names, filter_params, filter_params_function, strict=True
        ):
            check_f = check_config_params(f_params_func)
            if check_f:
                apply_filters_params.append(f_params(**f_params_func))
                apply_filters_names.append(f_name)

        filters = FiltersParameters(
            filters=apply_filters_names, filter_params=apply_filters_params
        )

        findpeaks = FindPeaksKwargs(**findpeaks)
        peaks = PeakDetectionParameters(**pulse, find_peaks_kwargs=findpeaks)
        resample = ResampleParameters(**resample)
        params = Params(prefilter, filters, peaks, resample)

        channels = np.array(channels)

        log.info(f"Loading data into dashboard: {filepath}")

        time_display = 1
        channels, channel_length = select_channels(
            channels,
            probe_selected_channels,
            d.sensorarray,
        )

        sliced_data, time_slice = ds.select_data(
            d.data, time_index, time_display, d.metadata.samplerate
        )
        index_time_start = int(time_slice[0] * d.metadata.samplerate)

        sliced_data = apply_filters(
            sliced_data, d.metadata.samplerate, prefilter, filters
        )

        colors = [*px.colors.qualitative.Light24, *px.colors.qualitative.Vivid]
        fig = subplots.make_subplots(
            rows=channel_length,
            shared_xaxes=True,
            shared_yaxes="all",
        )

        fig.add_traces(
            [
                go.Scattergl(
                    x=time_slice,
                    y=sliced_data[:, i],
                    name=f"{i}",
                    mode="lines",
                    line_color=colors[i],
                    # line_width=1.1,
                )
                for i in channels
            ],
            rows=list(np.arange(channel_length) + 1),
            cols=[1] * channel_length,
        )
        peaks = None
        if detect_pulses:
            output = detect_peaks_on_block(
                sliced_data, d.metadata.samplerate, prefilter, params
            )
            if output:
                fig.add_traces(
                    [
                        go.Scattergl(
                            x=output["centers"] / d.metadata.samplerate
                            + time_slice[0],
                            y=sliced_data[output["centers"], ch],
                            mode="markers",
                            marker_symbol="arrow",
                            marker_color="red",
                            marker_size=10,
                            name=f"Peaks {ch}",
                        )
                        for ch in channels
                    ],
                    rows=list(np.arange(channel_length) + 1),
                    cols=[1] * channel_length,
                )
            peaks = output

        fig.update_layout(
            showlegend=False,
            clickmode="event+select",
            autosize=True,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig, peaks
