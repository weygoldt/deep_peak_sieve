import nixio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from plotly import subplots
from scipy.signal import welch

from thunderpulse import ui_callbacks

from . import channel_selection as cs
from . import data_selection as ds


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
        # Input("peak_storage", "data"),
        Input("sw_notch_filter", "value"),
        Input("notch", "value"),
    )
    def update_graph_psd(
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
        # peaks
        sw_notch_filter,
        notch,
    ):
        if tabs:
            if not tabs == "tab_psd":
                fig = default_traces_figure()
                return fig
        if not filepath:
            fig = default_traces_figure()
            return fig

        DATA_PATH = filepath["data_path"]
        if not DATA_PATH:
            fig = default_traces_figure()
            return fig

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

        f, cx = welch(
            sliced_data,
            sample_rate,
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
        nix_file.close()

        return fig
