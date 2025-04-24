import nixio
import numpy as np
from dash import Input, Output


def callbacks_autoencoder(app):
    @app.callback(
        Output("spinner_autoencoder", "children"),
        Input("filepath", "data"),
        Input("bt_autoencoder", "n_clicks"),
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

            channels = np.arange(recording.shape[1])
            for ch in channels:
                try:
                    del block.data_arrays[f"waveform_channel_{ch}"]
                except KeyError:
                    continue

            for ch in channels:
                channel = block.data_arrays[f"peaks_channel_{ch}"][:]
                if np.any(channel + higher_index > recording.shape[0]):
                    overshoot_higher_index = np.min(
                        np.where(channel + higher_index > recording.shape[0])[0]
                    )

                    wf = recording[:, ch][
                        channel[:overshoot_higher_index, np.newaxis]
                        + np.arange(lower_index, higher_index)
                    ]
                else:
                    wf = recording[:, ch][
                        channel[:, np.newaxis] + np.arange(lower_index, higher_index)
                    ]

                block.create_data_array(
                    name=f"waveform_channel_{ch}",
                    array_type="waveform_channel_array",
                    data=wf, 
                    dtype=nixio.DataType.Int16
                )

            nix_file.close()
