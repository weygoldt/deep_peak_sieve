import nixio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from plotly import subplots

from .channel_selection import select_channels


def callback_waveforms(app):
    @app.callback(
        Output("waveforms", "figure"),
        Input("vis_tabs", "active_tab"),
        Input("channel_range_slider", "value"),
        Input("filepath", "data"),
        Input("probe", "selectedData"),
        Input("waveform_rand_num", "value"),
        Input("waveform_higher", "value"),
        Input("waveform_lower", "value"),
    )
    def update_graph_waveforms(
        tabs,
        channels,
        filepath,
        probe_selected_channels,
        wave_num,
        higher,
        lower,
    ):
        if tabs:
            if not tabs == "tab_waveforms":
                fig = default_waveforms_plot()
                return fig
        if not filepath:
            fig = default_waveforms_plot()
            return fig

        DATA_PATH = filepath["data_path"]
        if not DATA_PATH:
            fig = default_waveforms_plot()
            return fig
        if not wave_num:
            fig = default_waveforms_plot()
            return fig

        nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
        block = nix_file.blocks[0]
        section = nix_file.sections["recording"]
        sample_rate = float(section["samplerate"][0])

        if isinstance(channels, list):
            channels = np.array(channels)

        probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        channels, channel_length = select_channels(
            channels,
            probe_selected_channels,
            probe_frame,
        )

        fig = plot_waveforms(
            block,
            sample_rate,
            channels,
            channel_length,
            wave_num,
            higher,
            lower,
        )

        nix_file.close()

        return fig


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


def plot_waveforms(
    block,
    sample_rate,
    channels,
    channel_length,
    wave_num,
    higher,
    lower,
):
    block_names = "".join([n.name for n in block.data_arrays])
    if "waveform_channel_" not in block_names:
        fig = default_waveforms_plot()
        return fig

    fig = subplots.make_subplots(
        rows=channel_length,
        shared_xaxes=True,
        shared_yaxes="all",
    )
    time_slice = np.arange(lower / 1000, higher / 1000, 1 / sample_rate)
    colors = [*px.colors.qualitative.Light24, *px.colors.qualitative.Vivid]

    # select randome waveforms
    selection_index = {}
    for ch in channels:
        selection_index[ch] = []
        wf_selection_pool = block.data_arrays[f"waveform_channel_{ch}"].shape[
            0
        ]
        if wf_selection_pool < wave_num:
            selection_index[ch] = np.arange(wf_selection_pool)
        else:
            selection_index[ch] = np.random.choice(
                np.arange(wf_selection_pool), size=wave_num, replace=False
            )

    [
        fig.add_traces(
            [
                go.Scattergl(
                    x=time_slice,
                    y=block.data_arrays[f"waveform_channel_{ch}"][num],
                    name=f"{ch}",
                    mode="markers+lines",
                    line_color=colors[ch],
                    line_width=1,
                    opacity=0.8,
                )
                for num in selection_index[ch]
            ],
            rows=i + 1,
            cols=1,
        )
        for i, ch in enumerate(channels)
    ]

    fig.update_layout(
        showlegend=False,
        clickmode="event+select",
        autosize=True,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    [
        fig.update_yaxes(range=[-200, 50], row=i + 1, col=1)
        for i, _ in enumerate(channels)
    ]
    return fig
