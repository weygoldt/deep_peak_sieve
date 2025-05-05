import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, Patch
from IPython import embed

from thunderpulse.data_handling.data import load_data


def default_sensory_array_figure():
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


def callbacks_sensory_array(app):
    @app.callback(
        Output("probe", "figure"),
        Input("filepath", "data"),
        Input("channel_range_slider", "value"),
    )
    def plot_sensory_array(filepaths, channels):
        if not filepaths:
            return default_sensory_array_figure()

        data_path = filepaths["data_path"]
        if not data_path:
            return default_sensory_array_figure()

        d = load_data(**filepaths)
        sorted_sensoryarray_y = np.argsort(d.sensorarray.y)

        if np.array(channels).size > 2:
            patched_figure = Patch()
            colors = np.array(["blue"] * d.sensorarray.ids.shape[0])

            colors[sorted_sensoryarray_y[channels]] = "red"

            patched_figure["data"][0]["marker"]["color"] = colors.tolist()

            return patched_figure

        padding = 30
        x_range = [
            np.min(d.sensorarray.x) - padding,
            np.max(d.sensorarray.x) + padding,
        ]
        y_range = [
            np.min(d.sensorarray.y) - padding,
            np.max(d.sensorarray.y) + padding,
        ]
        colors = np.array(["blue"] * len(d.sensorarray.ids))

        if np.array(channels).size == 2:
            print(channels)
            colors[
                sorted_sensoryarray_y[np.arange(channels[0], channels[1] + 1)]
            ] = "red"
        else:
            colors[sorted_sensoryarray_y[channels[0]]] = "red"

        fig = go.Figure(
            data=go.Scattergl(
                x=d.sensorarray.x,
                y=d.sensorarray.y,
                mode="markers",
                marker_symbol="square",
                marker_color=colors,
                marker_sizemode="area",
                hoverinfo="text",
                hovertemplate="<b>%{text}</b>",
                text=[f"Channel index {int(i)}" for i in d.sensorarray.ids],
            ),
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_range=x_range,
            yaxis_range=y_range,
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event+select",
        )
        return fig
