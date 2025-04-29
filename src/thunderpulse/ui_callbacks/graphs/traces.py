import pathlib

from IPython import embed
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from plotly import subplots

from thunderpulse.data_handling.data import load_data
from thunderpulse.data_handling.preprocessing import (
    preprocessing_current_slice,
)
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
        Output("traces", "figure"),
        Output("peak_storage", "data"),
        Input("vis_tabs", "active_tab"),
        Input("time_slider", "value"),
        Input("channel_range_slider", "value"),
        Input("sw_bandpass_filter", "value"),
        Input("lowcutoff", "value"),
        Input("highcutoff", "value"),
        Input("sw_common_reference", "value"),
        Input("filepath", "data"),
        Input("probe", "selectedData"),
        Input("sw_peaks_current_window", "value"),
        Input("n_median", "value"),
        Input("sw_processed", "value"),
        Input("exclude_radius", "value"),
        Input("sw_merged_peaks", "value"),
        Input("sw_notch_filter", "value"),
        Input("notch", "value"),
        Input("threshold_artefact", "value"),
    )
    def update_graph_traces(
        tabs,
        time_index: int,
        channels,
        switch_bandpass,
        low,
        high,
        switch_common_reference,
        filepath,
        probe_selected_channels,
        sw_peak_detection,
        n_median,
        sw_processed,
        exclude_radius,
        sw_merged_peaks,
        sw_notch_filter,
        notch,
        th_artefact,
    ):
        if tabs:
            if not tabs == "tab_traces":
                fig = default_traces_figure()
                return fig, None
        if not filepath:
            fig = default_traces_figure()
            return fig, None

        data_path = pathlib.Path(filepath["data_path"])
        if not data_path:
            fig = default_traces_figure()
            return fig, None

        if isinstance(channels, list):
            channels = np.array(channels)

        log.info(f"Loading data into dashboard: {filepath}")
        d = load_data(**filepath)
        time_display = 1

        # probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        # channels, channel_length = cs.select_channels(
        #     channels,
        #     probe_selected_channels,
        #     probe_frame,
        # )
        # BUG: HARD CODED needs to be dynamic from probe graph
        # and channel selector

        if channels.size == 1:
            channels = np.append(channels, channels[0])
        channel_length = np.arange(channels[0], channels[1]).shape[0] + 1
        if channels[0] == channels[1]:
            channel_length = 1
        channels = np.arange(channels[0], channels[1] + 1)

        # channels = np.arange(channels)
        # channel_length = len(channels)

        sliced_data, time_slice = ds.select_data(
            d.data, time_index, time_display, d.metadata.samplerate
        )
        index_time_start = int(time_slice[0] * d.metadata.samplerate)

        # if sw_processed:
        #     recording = nix_file.blocks[0].data_arrays["processed_data"]
        #     sliced_data, time_slice = ds.select_data(
        #         recording, time_index, time_display, sample_rate
        #     )
        # else:
        sliced_data = preprocessing_current_slice(
            sliced_data,
            d.metadata.samplerate,
            switch_bandpass,
            low,
            high,
            switch_common_reference,
            sw_notch_filter,
            notch,
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
        peaks_ = None
        # if sw_peak_detection:
        #     peaks = processing.peak_detection.peaks_current_slice(
        #         sliced_data, index_time_start, channels, n_median, th_artefact
        #     )
        #
        #     fig.add_traces(
        #         [
        #             go.Scattergl(
        #                 x=peaks[peaks["channel"] == i]["spike_index"]
        #                 / d.metadata.samplerate,
        #                 y=peaks[peaks["channel"] == i]["amplitude"],
        #                 mode="markers",
        #                 marker_symbol="arrow",
        #                 marker_color="red",
        #                 marker_size=10,
        #                 name=f"Peaks {i}",
        #             )
        #             for i in channels
        #         ],
        #         rows=list(np.arange(channel_length) + 1),
        #         cols=[1] * channel_length,
        #     )
        #
        #     peaks_ = dict(
        #         index=np.arange(peaks.size),
        #         spike_index=peaks["spike_index"],
        #         amplitude=np.round(
        #             peaks["amplitude"],
        #             4,
        #         ),
        #         channel=peaks["channel"],
        #     )
        #     if sw_merged_peaks:
        #         if not exclude_radius:
        #             peaks_excluded = np.array([])
        #         else:
        #             peaks_without, peaks_excluded = (
        #                 processing.peak_detection.exclude_peaks_with_distance_traces(
        #                     peaks, probe_frame, exclude_radius
        #                 )
        #             )
        #
        #         if peaks_excluded.size > 0:
        #             fig.add_traces(
        #                 [
        #                     go.Scattergl(
        #                         x=peaks_excluded[
        #                             peaks_excluded["channel"] == i
        #                         ]["spike_index"]
        #                         / d.metadata.samplerate,
        #                         y=peaks_excluded[
        #                             peaks_excluded["channel"] == i
        #                         ]["amplitude"],
        #                         mode="markers",
        #                         marker_symbol="arrow",
        #                         marker_color="blue",
        #                         marker_size=10,
        #                         name=f"Peaks {i}",
        #                     )
        #                     for i in channels
        #                 ],
        #                 rows=list(np.arange(channel_length) + 1),
        #                 cols=[1] * channel_length,
        #             )

        fig.update_layout(
            showlegend=False,
            clickmode="event+select",
            autosize=True,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        # nix_file.close()

        return fig, peaks_
