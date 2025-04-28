import nixio
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, Patch


def default_probe_figure():
    fig = go.Figure(
        data=go.Scattergl(),
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_range=[-200, 200],
        yaxis_range=[-10, 500],
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
    )
    return fig


def callbacks_probe(app):
    @app.callback(
        Output("probe", "figure"),
        Input("filepath", "data"),
        Input("channel_range_slider", "value"),
    )
    def plot_probe(filepaths, channels):
        if not filepaths:
            fig = default_probe_figure()
            return fig

        DATA_PATH = filepaths["data_path"]
        if not DATA_PATH:
            fig = default_probe_figure()
            return fig

        nix_file = nixio.File(filepaths["data_path"], nixio.FileMode.ReadOnly)
        probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        sorted_probeframe_y = np.argsort(probe_frame["y"])
        if np.array(channels).size > 2:
            patched_figure = Patch()
            colors = np.array(["blue"] * probe_frame.shape[0])
            colors[
                sorted_probeframe_y[np.arange(channels[0], channels[-1] + 1)]
            ] = "red"
            patched_figure["data"][0]["marker"]["color"] = colors.tolist()

            return patched_figure

        colors = np.array(["blue"] * probe_frame.shape[0])
        colors[
            sorted_probeframe_y[np.arange(channels[0], channels[1] + 1)]
        ] = "red"
        fig = go.Figure(
            data=go.Scattergl(
                x=probe_frame["x"],
                y=probe_frame["y"],
                mode="markers",
                marker_symbol="square",
                marker=dict(color=colors),
                marker_sizemode="area",
                hoverinfo="text",
                hovertemplate="<b>%{text}</b>",
                text=[
                    f"Channel index {int(i) - 1}"
                    for i in probe_frame["contact_ids"]
                ],
            ),
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_range=[-200, 200],
            yaxis_range=[-10, 500],
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event+select",
        )
        nix_file.close()
        return fig
