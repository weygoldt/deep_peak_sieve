import nixio
import numpy as np
import numpy.typing as npt


def select_channels(
    channels: npt.NDArray,
    probe_selected_channels: dict | None,
    probe_frame: nixio.DataFrame,
):
    if channels.size == 1:
        channels = np.append(channels, channels[0])

    channel_length = np.arange(channels[0], channels[1]).shape[0] + 1

    if channels[0] == channels[1]:
        channel_length = 1
    channels = np.arange(channels[0], channels[1] + 1)

    sorted_after_y_pos = np.argsort(probe_frame["y"])
    channels = (
        probe_frame["contact_ids"][sorted_after_y_pos][channels].astype(
            np.int32
        )
        - 1
    )

    if probe_selected_channels:
        if probe_selected_channels["points"]:
            channels = np.zeros(
                len(probe_selected_channels["points"]), dtype=np.int32
            )
            for i, items in enumerate(probe_selected_channels["points"]):
                channel_id = items["text"].split(" ")[-1]
                channels[i] = int(channel_id)

            order = {key: i for i, key in enumerate(sorted_after_y_pos)}
            channels = np.array(sorted(channels, key=lambda d: order[d]))
            channel_length = len(channels)

    return channels.astype(np.int32)[::-1], channel_length
