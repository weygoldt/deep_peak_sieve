import nixio
from dash import Input, Output, dcc, html


def create_time_slider():
    time_slider = html.Div(
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
    return time_slider


def callback_time_slider(app):
    @app.callback(
        Output("time_slider_text", "children"),
        Output("time_slider", "max"),
        Output("time_slider", "step"),
        Input("filepath", "data"),
    )
    def inital_time_slice(filepath):
        if not filepath:
            return "Time Slider [1s]", 10, 1
        if not filepath["data_path"]:
            return "Time Slider [1 s]", 10, 1

        nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
        data_shape = nix_file.blocks[0].data_arrays["data"].shape[0]
        section = nix_file.sections["recording"]
        sample_rate = float(section["samplerate"][0])
        return "Time Slider [1 s]", data_shape, sample_rate * 0.5
