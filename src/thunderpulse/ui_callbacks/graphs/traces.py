import numpy as np
import plotly.colors
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
    PostProcessingParameters,
    PreProcessingParameters,
    SavgolParameters,
)
from thunderpulse.pulse_detection.detection import (
    apply_filters,
    detect_peaks_on_block,
)
from thunderpulse.ui_callbacks.graphs.channel_selection import select_channels
from thunderpulse.utils.check_config import check_config_params
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
            "pulse_detection_config": Input("pulse_detection_config", "data"),
        },
    )
    def update_graph_traces(general, pulse_detection_config):
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

        params = Params.from_dict(pulse_detection_config)

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
        sliced_data = apply_filters(sliced_data, d.metadata.samplerate, params)

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
            blockinfo = {
                "blockiterval": 0,
                "blocksize": int(time_display * d.metadata.samplerate),
                "overlap": 0,
            }
            output = detect_peaks_on_block(
                sliced_data,
                d.metadata.samplerate,
                blockinfo,
                params,
            )

            if output:
                unique_groups = sorted(set(output["groub"]))
                color_set = plotly.colors.qualitative.Dark24
                n_colors = len(color_set)
                group_to_color = {
                    group: color_set[i % n_colors]
                    for i, group in enumerate(unique_groups)
                }
                for i, ch in enumerate(channels, start=1):
                    pulse_index = output["channels"] == ch
                    pulses = output["centers"][pulse_index]
                    groubs = output["groub"][pulse_index]
                    pulse_colors = [group_to_color[g] for g in groubs]
                    fig.add_trace(
                        go.Scattergl(
                            x=pulses / d.metadata.samplerate + time_slice[0],
                            y=sliced_data[pulses, ch],
                            mode="markers",
                            marker_symbol="arrow",
                            marker_color=pulse_colors,
                            marker_size=10,
                            name=f"Peaks {ch}",
                        ),
                        row=i,
                        col=[1] * channel_length,
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
