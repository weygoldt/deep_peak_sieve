import nixio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# import ruptures as rpt
from dash import Input, Output
from IPython import embed
from joblib import Parallel, delayed
from plotly import subplots

from . import channel_selection as cs


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


def callbacks_spikes_ampl(app):
    @app.callback(
        Output("spikes_ampl", "figure"),
        Input("vis_tabs", "active_tab"),
        Input("filepath", "data"),
        Input("channel_range_slider", "value"),
        Input("probe", "selectedData"),
        # Input("num_windowsize_ampl", "value"),
        # Input("bt_save_mean_spike_ampl", "n_clicks"),
    )
    def update_graph_spikes_ampl(
        tabs,
        filepath,
        channels,
        probe_selected_channels,  # win_size, bt_mean_spikes_ampl
    ):
        if tabs:
            if not tabs == "tab_spikes_ampl":
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
        try:
            recording = nix_file.blocks[0].data_arrays["processed_data"]
        except KeyError:
            fig = default_traces_figure()
            return fig

        section = nix_file.sections["recording"]
        sample_rate = float(section["samplerate"][0])

        if isinstance(channels, list):
            channels = np.array(channels)

        probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        channels, channel_length = cs.select_channels(
            channels,
            probe_selected_channels,
            probe_frame,
        )

        spike_frame = nix_file.blocks[0].data_frames["spike_times_dataframe_processed"]

        colors = [*px.colors.qualitative.Light24, *px.colors.qualitative.Vivid]

        data_arrays = nix_file.blocks[0].data_arrays
        filterd_ch = [
            fch
            for fch in data_arrays
            if fch.type == "ampl" and int(fch.name.split("_")[-1]) in channels
        ]
        idex_ch = np.array(
            [np.where(ch == np.sort(channels))[0] for ch in channels]
        ).flatten()

        filterd_ch_all = [fch for fch in data_arrays if fch.type == "ampl"]
        all_seps = [fch for fch in data_arrays if "seperations_channel" in fch.name]

        all_seps_flatten = []
        for ch in np.arange(32):
            seperation_index_channel = all_seps[ch][:]
            spikes_channel = (
                spike_frame.read_rows(spike_frame["channel"] == ch)["spike_index"]
                / sample_rate
            )
            if not np.all(seperation_index_channel) > 0:
                continue
            if seperation_index_channel[-1] >= spikes_channel.size:
                seperation_index_channel[-1] = spikes_channel.size - 1
                all_seps_flatten.extend(spikes_channel[seperation_index_channel])
            else:
                all_seps_flatten.extend(spikes_channel[seperation_index_channel])

        all_seps_flatten = np.sort(all_seps_flatten)

        fig = subplots.make_subplots(
            rows=channel_length + 1,
            shared_xaxes=True,
        )
        fig.add_traces(
            [
                go.Scattergl(
                    x=spike_frame.read_rows(spike_frame["channel"] == i)["spike_index"]
                    / sample_rate,
                    y=spike_frame.read_rows(spike_frame["channel"] == i)["amplitude"],
                    name=f"{i}",
                    mode="markers",
                    marker_color=colors[i],
                )
                for i in channels
            ],
            rows=list(np.arange(channel_length) + 1),
            cols=[1] * channel_length,
        )

        fig.add_traces(
            [
                go.Scattergl(
                    x=spike_frame.read_rows(spike_frame["channel"] == i)["spike_index"]
                    / sample_rate,
                    y=filterd_ch[ch][:],
                    name=f"{i}",
                    line_color="white",
                )
                for i, ch in zip(channels, idex_ch)
            ],
            rows=list(np.arange(channel_length) + 1),
            cols=[1] * channel_length,
        )

        for c, ch in enumerate(channels, start=1):
            seps_channel = [
                sep for sep in all_seps if int(sep.name.split("_")[-1]) == ch
            ][0][:]
            if not np.all(seps_channel) > 0:
                continue
            y_min = np.min(
                spike_frame.read_rows(spike_frame["channel"] == ch)["amplitude"]
            )
            y_max = np.max(
                spike_frame.read_rows(spike_frame["channel"] == ch)["amplitude"]
            )

            spikes_channel = (
                spike_frame.read_rows(spike_frame["channel"] == ch)["spike_index"]
                / sample_rate
            )
            if seps_channel[-1] >= spikes_channel.shape[0]:
                seps_channel[-1] = spikes_channel.size - 1

            for x in seps_channel:
                fig.add_trace(
                    go.Scattergl(
                        x=[
                            spikes_channel[x],
                            spikes_channel[x],
                        ],  # Same x for both points
                        y=[y_min, y_max],  # From bottom to top of subplot
                        mode="lines",
                        line=dict(color="white", width=2, dash="dash"),
                        showlegend=False,
                        hoverinfo="none",
                    ),
                    row=c,
                    col=1,
                )

        bins = np.arange(0, np.max(all_seps_flatten) + 1, 10)
        hist, bins = np.histogram(all_seps_flatten, bins)
        # bins = 0.5 * (bins[:-1] + bins[1:])
        fig.add_trace(
            go.Bar(
                x=bins,
                y=hist,
                marker_color="white",
            ),
            row=channel_length + 1,
            col=1,
        )
        threshold = np.where(hist >= 10)[0]
        edges = bins[threshold]
        for e in edges:
            fig.add_trace(
                go.Scattergl(
                    x=[e, e],
                    y=[0, np.max(hist)],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="none",
                ),
                row=channel_length + 1,
                col=1,
            )

        fig.update_layout(
            showlegend=False,
            clickmode="event+select",
            autosize=True,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        nix_file.close()

        return fig


# def calc_seperations(ch_data, ch):
#     algo_c = rpt.KernelCPD(kernel="linear", min_size=100).fit(ch_data)
#     res = algo_c.predict(pen=7000)
#     return res
