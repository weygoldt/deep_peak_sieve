import numpy as np
from dash import dcc, html


def create_channel_slider():
    channel_slider = html.Div(
        [
            html.H5(
                children="Channel Selector", style={"textAlign": "center"}
            ),
            dcc.RangeSlider(
                0,
                10,
                1,
                id="channel_range_slider",
                tooltip={"placement": "bottom"},
                value=np.array([0, 15]),
                count=1,
            ),
        ]
    )
    return channel_slider
