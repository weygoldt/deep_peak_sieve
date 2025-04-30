from dash import dcc, html


def create_time_slider():
    return html.Div(
        [
            html.H5(
                children="Time Slider [1 s]",
                id="time_slider_text",
                style={"textAlign": "center"},
            ),
            dcc.Slider(
                0,
                10,
                1,
                id="time_slider",
                marks=None,
                updatemode="drag",
                value=0,
                tooltip={"placement": "bottom"},
            ),
        ]
    )
