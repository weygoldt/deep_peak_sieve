from IPython import embed
import nixio
import numpy as np
from rich.progress import track
from dash import Input, Output


def callbacks_create_waveforms(app):
    @app.callback(
        Output("spinner_waveforms", "children"),
        Input("filepath", "data"),
        Input("bt_save_waveforms", "n_clicks"),
        Input("waveform_higher", "value"),
        Input("waveform_lower", "value"),
    )
    def create_wave_form(filepath, n_clicks, higher, lower):
        if n_clicks and n_clicks > 0:
            nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadWrite)
            block = nix_file.blocks[0]
            recording = block.data_arrays["processed_data"]
            section = nix_file.sections["recording"]
            sample_rate = float(section["samplerate"][0])
            lower_index = int(sample_rate * (lower / 1000))
            higher_index = int(sample_rate * (higher / 1000))
            spike_frame = block.data_frames["spike_times_dataframe_processed"]

            channels = np.arange(recording.shape[1])
            for ch in channels:
                try:
                    del block.data_arrays[f"waveform_channel_{ch}"]
                except KeyError:
                    continue

            try:
                spike_frame.append_column(
                    np.ones(spike_frame.shape[0], dtype=bool), "wf"
                )
            except ValueError:
                pass

            # for ch in channels:
            for ch in track(channels, description="Calc channels"):
                spikes_channel = spike_frame.read_rows(spike_frame["channel"] == ch)[
                    "spike_index"
                ]
                channel_index = np.where(spike_frame["channel"] == ch)[0]

                if np.any(spikes_channel + higher_index > recording.shape[0]):
                    overshoot_higher_index = np.min(
                        np.where(spikes_channel + higher_index > recording.shape[0])[0]
                    )
                    index_overshoot = len(spikes_channel) - overshoot_higher_index

                    cells = spike_frame[channel_index[-index_overshoot:]]
                    cells["wf"] = False
                    spike_frame.write_rows(cells, channel_index[-index_overshoot:])

                    wf = recording[:, ch][
                        spikes_channel[:overshoot_higher_index, np.newaxis]
                        + np.arange(lower_index, higher_index)
                    ]
                else:
                    wf = recording[:, ch][
                        spikes_channel[:, np.newaxis]
                        + np.arange(lower_index, higher_index)
                    ]

                block.create_data_array(
                    name=f"waveform_channel_{ch}",
                    array_type="waveform_channel_array",
                    data=wf,
                    dtype=nixio.DataType.Int16,
                )

            nix_file.close()
