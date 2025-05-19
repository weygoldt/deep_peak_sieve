import nixio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from IPython import embed
from plotly import subplots

from thunderpulse.data_handling.data import load_data
from thunderpulse.pulse_detection.config import Params
from thunderpulse.pulse_detection.detection import detect_peaks_on_block
from thunderpulse.ui_callbacks.graphs.channel_selection import select_channels
from thunderpulse.ui_callbacks.graphs.data_selection import select_data


def default_waveforms_plot():
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


def callbacks(app):
    @app.callback(
        Output("waveforms", "figure"),
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
                "vis_tabs": Input("vis_tabs", "active_tab"),
                "time_slider": Input("time_slider", "value"),
                "channels": Input("channel_range_slider", "value"),
                "probe_selected_channels": Input("probe", "selectedData"),
            },
            "pulse_detection_config": Input("pulse_detection_config", "data"),
        },
    )
    def update_graph_waveforms(
        general: dict,
        pulse_detection_config,
    ):
        (
            filepath,
            tabs,
            time_index,
            channels,
            probe_selected_channels,
        ) = general.values()

        if tabs and tabs != "tab_waveforms":
            return default_waveforms_plot()
        if not filepath:
            return default_waveforms_plot()
        if not filepath["data_path"]:
            return default_waveforms_plot()

        d = load_data(**filepath)

        params = Params.from_dict(pulse_detection_config)

        channels = np.array(channels)

        channels, channel_length = select_channels(
            channels,
            probe_selected_channels,
            d.sensorarray,
        )

        fig = subplots.make_subplots(
            rows=channel_length,
            shared_xaxes=True,
            shared_yaxes="all",
        )
        time_display = 1.0

        blockinfo = {
            "blockiterval": 0,
            "blocksize": int(1.0 * d.metadata.samplerate),
            "overlap": 0,
        }
        sliced_data, time_slice = select_data(
            d.data, time_index, time_display, d.metadata.samplerate
        )
        pulse_storage = detect_peaks_on_block(
            sliced_data,
            d.metadata.samplerate,
            blockinfo,
            params,
        )
        if not pulse_storage:
            return default_waveforms_plot()

        colors = [*px.colors.qualitative.Light24, *px.colors.qualitative.Vivid]
        for i, ch in enumerate(channels, start=1):
            pulse_index = pulse_storage["channels"] == ch
            pulses = np.array(pulse_storage["pulses"])[pulse_index]

            for p in pulses:
                fig.add_trace(
                    go.Scattergl(
                        x=np.arange(p.shape[0]) / d.metadata.samplerate
                        - params.peaks.cutout_window_around_peak_s,
                        y=p,
                        mode="lines",
                        line_color=colors[ch],
                        name=f"Peaks {ch}",
                        line_width=1,
                        opacity=0.8,
                    ),
                    row=i,
                    col=[1] * channel_length,
                )

        fig.update_layout(
            showlegend=False,
            clickmode="event+select",
            autosize=True,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig
