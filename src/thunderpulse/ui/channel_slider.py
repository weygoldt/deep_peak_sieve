from IPython import embed
from dash import html, dcc, Output, Input
import numpy as np
import nixio


def create_channel_slider():
    channel_slider = html.Div(

        [
            html.H5(children="Channel Selector", style={"textAlign": "center"}),
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


def callback_channel_slider(app):
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
        else:
            nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
            section = nix_file.sections["recording"]
            channels = int(section["channels"])
            probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
            sorted_after_y_pos = np.argsort(probe_frame["y"])

            nix_file.close()
            marks = {
                f"{id[0]}": {"label": f"{id[1]}"}
                for id in zip(np.arange(channels), sorted_after_y_pos)
            }
            return channels - 1, marks

    @app.callback(
        Output("channel_range_slider", "value"),
        Input("probe", "selectedData"),
        Input("filepath", "data"),
    )
    def update_channels(selected_data, filepaths):
        if not selected_data:
            return [0, 15]
        elif not selected_data["points"]:
            return [0, 15]
        else:
            nix_file = nixio.File(filepaths["data_path"], nixio.FileMode.ReadOnly)
            probe_frame = nix_file.blocks[0].data_frames["probe_frame"]
            sorted_after_y_pos = np.argsort(probe_frame["y"])
            channels = []
            for items in selected_data["points"]:
                channel_id = items["text"].split(" ")[-1]
                channels.append(int(channel_id))

            order = {key: i for i, key in enumerate(sorted_after_y_pos)}
            channels = np.array(sorted(channels, key=lambda d: order[d]))

            start_channel = np.where(channels[0] == sorted_after_y_pos)[0]
            stop_channel = np.where(channels[-1] == sorted_after_y_pos)[0]

            channels = np.arange(start_channel.item(), stop_channel.item() + 1)
            nix_file.close()
            return channels
