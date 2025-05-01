from dash import Input, Output

from thunderpulse.data_handling.data import load_data


def callbacks(app):
    @app.callback(
        Output("time_slider_text", "children"),
        Output("time_slider", "max"),
        Output("time_slider", "step"),
        Input("filepath", "data"),
    )
    def inital_time_slice(filepath):
        if not filepath:
            return (
                "Time Slider [1s]",
                10,
                1,
            )
        if not filepath["data_path"]:
            return (
                "Time Slider [1 s]",
                10,
                1,
            )
        ds = load_data(**filepath)
        return (
            "Time Slider [1 s]",
            ds.metadata.frames,
            ds.metadata.samplerate * 0.5,
        )
