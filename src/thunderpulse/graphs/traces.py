from IPython import embed
from dash import Output, Input
import numpy as np
import plotly.graph_objects as go
from plotly import subplots
import plotly.express as px
import nixio

from . import channel_selection as cs
from . import data_selection as ds

import thunderpulse.processing as processing


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

        DATA_PATH = filepath["data_path"]
        if not DATA_PATH:
            fig = default_traces_figure()
            return fig, None

        nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
        recording = nix_file.blocks[0].data_arrays["data"]
        section = nix_file.sections["recording"]
        sample_rate = float(section["samplerate"][0])
        time_display = 1

        if isinstance(channels, list):
            channels = np.array(channels)

        probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        channels, channel_length = cs.select_channels(
            channels,
            probe_selected_channels,
            probe_frame,
        )

        sliced_data, time_slice = ds.select_data(
            recording, time_index, time_display, sample_rate
        )
        index_time_start = int(time_slice[0] * sample_rate)

        if sw_processed:
            recording = nix_file.blocks[0].data_arrays["processed_data"]
            sliced_data, time_slice = ds.select_data(
                recording, time_index, time_display, sample_rate
            )
        else:
            sliced_data = processing.preprocessing.preprocessing_current_slice(
                sliced_data,
                sample_rate,
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
                    line_color=colors[i]
                    # line_width=1.1,
                )
                for i in channels
            ],
            rows=list(np.arange(channel_length) + 1),
            cols=[1] * channel_length,
        )
        peaks_ = None
        if sw_peak_detection:
            peaks = processing.peak_detection.peaks_current_slice(
                sliced_data, index_time_start, channels, n_median, th_artefact
            )

            fig.add_traces(
                [
                    go.Scattergl(
                        x=peaks[peaks["channel"] == i]["spike_index"] / sample_rate,
                        y=peaks[peaks["channel"] == i]["amplitude"],
                        mode="markers",
                        marker_symbol="arrow",
                        marker_color="red",
                        marker_size=10,
                        name=f"Peaks {i}",
                    )
                    for i in channels
                ],
                rows=list(np.arange(channel_length) + 1),
                cols=[1] * channel_length,
            )

            peaks_ = dict(
                index=np.arange(peaks.size),
                spike_index=peaks["spike_index"],
                amplitude=np.round(
                    peaks["amplitude"],
                    4,
                ),
                channel=peaks["channel"],
            )
            if sw_merged_peaks:
                if not exclude_radius:
                    peaks_excluded = np.array([])
                else:
                    peaks_without, peaks_excluded = (
                        processing.peak_detection.exclude_peaks_with_distance_traces(
                            peaks, probe_frame, exclude_radius
                        )
                    )

                if peaks_excluded.size > 0:
                    fig.add_traces(
                        [
                            go.Scattergl(
                                x=peaks_excluded[peaks_excluded["channel"] == i][
                                    "spike_index"
                                ]
                                / sample_rate,
                                y=peaks_excluded[peaks_excluded["channel"] == i][
                                    "amplitude"
                                ],
                                mode="markers",
                                marker_symbol="arrow",
                                marker_color="blue",
                                marker_size=10,
                                name=f"Peaks {i}",
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
        )
        nix_file.close()

        return fig, peaks_
