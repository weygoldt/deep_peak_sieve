import warnings

import nixio
import numpy as np
import umap
from dash import Input
from joblib import Parallel, delayed
from rich.progress import track
from sklearn.cluster import HDBSCAN

warnings.filterwarnings("ignore")


def callbacks_create_umap_embedding(app):
    @app.callback(
        Input("filepath", "data"),
        Input("bt_umap_embedding", "n_clicks"),
        Input("waveform_higher", "value"),
        Input("waveform_lower", "value"),
    )
    def create_wave_form(filepath, n_clicks, higher, lower):
        if n_clicks and n_clicks > 0:
            reducer = umap.UMAP()
            nix_file = nixio.File(
                filepath["data_path"], nixio.FileMode.ReadWrite
            )
            block = nix_file.blocks[0]
            recording = block.data_arrays["processed_data"]
            section = nix_file.sections["recording"]
            sample_rate = float(section["samplerate"][0])
            spike_frame = block.data_frames["spike_times_dataframe_processed"]

            channels = np.arange(recording.shape[1])
            for ch in channels:
                try:
                    del block.data_arrays[f"umap_channel_{ch}"]
                except KeyError:
                    continue

            res = Parallel(n_jobs=1)(
                delayed(multprocess_umap)(filepath, ch, reducer)
                for ch in track(channels, description="Calc UMAP channel")
            )
            # res = Parallel(n_jobs=1)(
            #     delayed(multprocess_umap)(filepath, ch, reducer) for ch in channels
            # )
            # for ch in progress:
            #     # for ch in channels:
            #     wf = block.data_arrays[f"waveform_channel_{ch}"]
            #     wf = preprocessing_wf(wf, block, ch)
            #     embedding = reducer.fit_transform(wf, ensure_all_finite=True)
            #     hdb = HDBSCAN(min_cluster_size=50, n_jobs=-1)
            #     labels = hdb.fit_predict(embedding)
            #     segments = spike_frame.read_rows(spike_frame["channel"] == ch)[
            #         "segment"
            #     ][: len(labels)]
            #     emb = np.hstack(
            #         (embedding, labels.reshape(-1, 1), segments.reshape(-1, 1))
            #     )
            for ch, emb in enumerate(res):
                block.create_data_array(
                    name=f"umap_channel_{ch}",
                    array_type="[umap, labels, cluster, segments]",
                    data=emb[0],
                )
                try:
                    del block.data_arrays[f"waveform_channel_processed_{ch}"]
                    block.create_data_array(
                        name=f"waveform_channel_processed_{ch}",
                        array_type="np.ndarray",
                        data=emb[1],
                    )
                except KeyError:
                    block.create_data_array(
                        name=f"waveform_channel_processed_{ch}",
                        array_type="np.ndarray",
                        data=emb[1],
                    )
            nix_file.close()


def preprocessing_wf(wf, block, ch):
    # wf_processed = signal.resample(wf[:], wf.shape[1]*5, axis=1)
    # new_wf_time=
    # wf_processed = np.interp(
    min_peaks = np.argmin(wf[:], axis=1)
    # After upsampling and peak detection
    n_samples = wf.shape[1]
    center = n_samples // 2
    # Create shifted indices using broadcasting
    shifts = center - min_peaks
    indices = (np.arange(n_samples) - shifts[:, None]) % n_samples
    # Advanced indexing for vectorized alignment
    aligned_wf = wf[:][np.arange(len(shifts))[:, None], indices]

    return aligned_wf


def multprocess_umap(filepath, ch, reducer):
    nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
    block = nix_file.blocks[0]
    spike_frame = block.data_frames["spike_times_dataframe_processed"]
    wf = block.data_arrays[f"waveform_channel_{ch}"]
    wf_processed = preprocessing_wf(wf, block, ch)
    if wf_processed.shape[0] > 2:
        embedding = reducer.fit_transform(wf_processed, ensure_all_finite=True)
        hdb = HDBSCAN(min_cluster_size=25, n_jobs=-1)
        labels = hdb.fit_predict(embedding)
        segments = spike_frame.read_rows(spike_frame["channel"] == ch)[
            "segment"
        ][: len(labels)]
        emb = np.hstack(
            (embedding, labels.reshape(-1, 1), segments.reshape(-1, 1))
        )
    else:
        segments = spike_frame.read_rows(spike_frame["channel"] == ch)[
            "segment"
        ]
        emb = np.hstack(
            (
                np.zeros((segments.shape[0], 2)) - 1,
                np.zeros_like(segments).reshape(-1, 1) - 1,
                segments.reshape(-1, 1),
            )
        )
    nix_file.close()
    return emb, wf_processed
