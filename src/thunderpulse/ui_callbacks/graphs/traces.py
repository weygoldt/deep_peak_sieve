import pathlib
import re

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from IPython import embed
from nixio import file
from plotly import subplots

import thunderpulse.ui_callbacks.graphs.channel_selection as cs
from thunderpulse.data_handling.data import load_data
from thunderpulse.pulse_detection.config import (
    BandpassParameters,
    FiltersParameters,
    NotchParameters,
    Params,
    PeakDetectionParameters,
    PrefilterParameters,
    ResampleParameters,
    SavgolParameters,
)
from thunderpulse.pulse_detection.detection import apply_filters
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
        Output("traces", "figure"),
        # Filter
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
                "vis_tabs": Input("vis_tabs", "active_tab"),
                "time_slider": Input("time_slider", "value"),
                "channels": Input("channel_range_slider", "value"),
                "probe_selected_channels": Input("probe", "selectedData"),
            },
            "pre_filter": {
                "common_median_reference": Input(
                    "sw_common_reference", "value"
                ),
            },
            "savgol": {
                "window_length": Input("num_savgol_window_length", "value"),
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
        filepath, tabs, time_index, channels, probe_selected_channels = (
            general.values()
        )
        if tabs and tabs != "tab_traces":
            return default_traces_figure()
        if not filepath:
            return default_traces_figure()
        if not filepath["data_path"]:
            return default_traces_figure()

        prefilter = PrefilterParameters(**pre_filter)
        filters = FiltersParameters(
            filters=["savgol", "bandpass", "notch"],
            filter_params=[
                SavgolParameters(**savgol),
                BandpassParameters(**bandpass),
                NotchParameters(**notch),
            ],
        )
        peaks = PeakDetectionParameters(**pulse, find_peaks_kwargs=findpeaks)
        resample = ResampleParameters(**resample)
        params = Params(prefilter, filters, peaks, resample)

        channels = np.array(channels)

        log.info(f"Loading data into dashboard: {filepath}")
        d = load_data(**filepath)

        time_display = 1
        channels, channel_length = cs.select_channels(
            channels,
            probe_selected_channels,
            d.sensorarray,
        )

        sliced_data, time_slice = ds.select_data(
            d.data, time_index, time_display, d.metadata.samplerate
        )
        index_time_start = int(time_slice[0] * d.metadata.samplerate)

        sliced_data = apply_filters(
            sliced_data, d.metadata.samplerate, filters
        )
        # if sw_processed:
        #     recording = nix_file.blocks[0].data_arrays["processed_data"]
        #     sliced_data, time_slice = ds.select_data(
        #         recording, time_index, time_display, sample_rate
        #     )
        # else:
        # sliced_data = preprocessing_current_slice(
        #     sliced_data,
        #     d.metadata.samplerate,
        #     switch_bandpass,
        #     low,
        #     high,
        #     switch_common_reference,
        #     sw_notch_filter,
        #     notch,
        # )

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

        return fig
