import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from plotly import subplots
from scipy.signal import welch

from thunderpulse.data_handling.data import load_data
from thunderpulse.dsp.common_reference import common_median_reference
from thunderpulse.pulse_detection.config import Params
from thunderpulse.pulse_detection.detection import apply_filters
from thunderpulse.ui_callbacks.graphs.channel_selection import select_channels
from thunderpulse.ui_callbacks.graphs.data_selection import select_data
from thunderpulse.utils.loggers import get_logger
from thunderpulse.utils.logging_setup import setup_logging

log = get_logger(__name__)
setup_logging(log)


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


def callbacks_psd(app):
    @app.callback(
        Output("psd", "figure"),
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
                "vis_tabs": Input("vis_tabs", "active_tab"),
                "time_slider": Input("time_slider", "value"),
                "channels": Input("channel_range_slider", "value"),
                "probe_selected_channels": Input("probe", "selectedData"),
                "detect_pulses": Input("sw_detect_pulses", "value"),
            },
            "pulse_detection_config": Input("pulse_detection_config", "data"),
            "pulse_storage": Input("peak_storage", "data"),
        },
    )
    def update_graph_psd(general, pulse_detection_config, pulse_storage):
        (
            filepath,
            tabs,
            time_index,
            channels,
            probe_selected_channels,
            detect_pulses,
        ) = general.values()

        if tabs and tabs != "tab_psd":
            return default_traces_figure()
        if not filepath:
            return default_traces_figure()
        if not filepath["data_path"]:
            return default_traces_figure()

        d = load_data(**filepath)

        params = Params.from_dict(pulse_detection_config)

        channels = np.array(channels)

        log.info(f"Loading data into dashboard: {filepath}")

        time_display = 1
        channels, channel_length = select_channels(
            channels,
            probe_selected_channels,
            d.sensorarray,
        )

        sliced_data, time_slice = select_data(
            d.data, time_index, time_display, d.metadata.samplerate
        )
        index_time_start = int(time_slice[0] * d.metadata.samplerate)

        # PreFilter operations
        if getattr(params.preprocessing, "common_median_reference", False):
            log.debug("Take the common median average")
            sliced_data = common_median_reference(sliced_data)
        sliced_data = apply_filters(sliced_data, d.metadata.samplerate, params)

        f, cx = welch(
            sliced_data,
            d.metadata.samplerate,
            nperseg=2**10,
            axis=0,
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
                    x=f,
                    y=cx[:, i],
                    name=f"{i}",
                    mode="lines",
                    line_color=colors[i],
                    # log_y=True,
                )
                for i in channels
            ],
            rows=list(np.arange(channel_length) + 1),
            cols=[1] * channel_length,
        )

        fig.update_layout(
            showlegend=False,
            clickmode="event+select",
            autosize=True,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            # y_axis="log",
        )
        fig.update_yaxes(
            type="log",
        )

        return fig
