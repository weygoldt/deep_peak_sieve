import numpy as np
from dash import Input, Output
from IPython import embed

from thunderpulse.data_handling.data import load_data
from thunderpulse.ui_callbacks.graphs.channel_selection import select_channels


def callbacks(app):
    @app.callback(
        Output("channel_range_slider", "max"),
        Output("channel_range_slider", "marks"),
        Input("filepath", "data"),
    )
    def inital_channels(filepath):
        if not filepath:
            return 10, None
        if not filepath["data_path"]:
            return 10, None

        # nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
        # section = nix_file.sections["recording"]
        # channels = int(section["channels"])
        # probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
        # sorted_after_y_pos = np.argsort(probe_frame["y"])
        # nix_file.close()

        ds = load_data(**filepath)
        # WARNING: Does not sort anymore after probe layout

        marks = {
            f"{id[0]}": {"label": f"{id[1]}"}
            for id in zip(
                np.arange(ds.metadata.channels),
                np.arange(ds.metadata.channels),
                strict=False,
            )
        }
        return ds.metadata.channels - 1, marks

    @app.callback(
        Output("channel_range_slider", "value"),
        Input("probe", "selectedData"),
        Input("filepath", "data"),
    )
    def update_channels(selected_data, filepaths):
        if not selected_data or not selected_data["points"]:
            return [0, 15]

        d = load_data(**filepaths)

        sorted_sensoryarray_y = np.argsort(d.sensorarray.y)

        channels = np.zeros(len(selected_data["points"]), dtype=np.int32)
        for i, items in enumerate(selected_data["points"]):
            channel_id = items["text"].split(" ")[-1]
            channels[i] = int(channel_id)

        order = {key: i for i, key in enumerate(sorted_sensoryarray_y)}
        channels = np.array(sorted(channels, key=lambda d: order[d]))
        channel_index = np.zeros_like(channels)
        for i, ch in enumerate(channels):
            channel_index[i] = np.where(ch == sorted_sensoryarray_y)[0][0]

        return np.sort(channel_index)
