import copy
import logging
from pathlib import Path

import nixio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, ctx
from IPython import embed
from nixio.cmd.upgrade import nix
from nixio.exceptions import DuplicateName
from shapely import Point, Polygon
from sklearn.cluster import HDBSCAN

from thunderpulse.data_handling.data import load_data
from thunderpulse.utils.loggers import get_logger
from thunderpulse.utils.logging_setup import setup_logging

log = get_logger(__name__)
setup_logging(log)


def default_umap_figure():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode="event+select",
    )
    return fig


def callbacks_umap(app):
    @app.callback(
        Output("umap", "figure"),
        Input("vis_tabs", "active_tab"),
        Input("filepath", "data"),
        Input("select_umap_embedding", "value"),
        Input("num_hdbscan_cluster_size", "value"),
        Input("bt_run_hdbscan", "n_clicks"),
    )
    def create_umap_embeding(
        tabs,
        filepath,
        umap_embedding,
        cluster_size,
        bt_run_hdbscan,
    ):
        if tabs and tabs != "tab_umap":
            return default_umap_figure()
        if not filepath:
            return default_umap_figure()
        if not filepath["data_path"]:
            return default_umap_figure()

        d = load_data(**filepath)

        save_path = list(Path(d.paths.save_path).rglob("*pulses.*"))

        try:
            save_file = [p for p in save_path if p.suffix in [".nix", ".h5"]][
                0
            ]
        except IndexError:
            return default_umap_figure()

        if not save_file.exists:
            return default_umap_figure()
        nix_file = nixio.File(str(save_file), nixio.FileMode.ReadOnly)
        block = nix_file.blocks[0]
        if not umap_embedding:
            log.debug("No umap embedding selected, or found")
            return default_umap_figure()

        embedding = block.data_arrays[f"{umap_embedding}"]
        button = ctx.triggered_id == "bt_run_hdbscan"

        if not button:
            fig = go.Figure(
                data=go.Scattergl(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode="markers",
                )
            )

            # Update the layout
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=0, b=0),
                clickmode="event+select",
                showlegend=False,
                coloraxis=dict(
                    colorbar=dict(len=0.5, yanchor="bottom", xanchor="right")
                ),
            )

            nix_file.close()

            return fig

        if not cluster_size:
            hdb = HDBSCAN()
        else:
            hdb = HDBSCAN(min_cluster_size=cluster_size)

        labels = hdb.fit_predict(embedding[:])
        cluster = np.unique(labels)
        colors = px.colors.qualitative.Light24
        num_clusters = cluster.shape[0]
        colorscale_values = np.linspace(0, 1, num_clusters + 1)

        colorscale = []
        for i in np.arange(colorscale_values.shape[0] - 1):
            colorscale.append([colorscale_values[i], colors[i]])
            colorscale.append([colorscale_values[i + 1], colors[i]])

        fig = go.Figure(
            data=go.Scattergl(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    color=labels,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=dict(text="Clusters", side="right"),
                        tickvals=cluster,
                        ticktext=[str(int(label)) for label in cluster],
                    ),
                ),
            )
        )

        # Update the layout
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event+select",
            showlegend=False,
            coloraxis=dict(
                colorbar=dict(len=0.5, yanchor="bottom", xanchor="right")
            ),
        )

        nix_file.close()

        return fig

    @app.callback(
        Output("umap_selection", "figure"),
        Output("store_umap_selection", "data"),
        Input("filepath", "data"),
        Input("umap", "selectedData"),
        Input("umap", "figure"),
        Input("store_umap_selection", "data"),
        Input("select_umap_embedding", "value"),
    )
    def plot_selected_umap_embedding(
        filepath, data, figure, saved_selections, umap_embedding
    ):
        if not data:
            fig = default_umap_figure()
            return fig, None

        try:
            figure["layout"]["selections"]
        except KeyError:
            fig = default_umap_figure()
            return fig, None

        umap_selections = {
            f"{i}": s
            for i, s in enumerate(figure["layout"]["selections"])
            if figure["layout"]["selections"]
        }
        if not saved_selections:
            saved_selections = copy.deepcopy(umap_selections)

        d = load_data(**filepath)

        save_path = list(Path(d.paths.save_path).rglob("*pulses.*"))
        save_file = [p for p in save_path if p.suffix in [".nix", ".h5"]][0]
        if not save_file.exists():
            return default_umap_figure()

        nix_file = nixio.File(str(save_file), nixio.FileMode.ReadOnly)
        block = nix_file.blocks[0]

        if not umap_embedding:
            log.debug("No umap embedding selected, or found")
            return default_umap_figure()

        embedding = block.data_arrays[f"{umap_embedding}"]
        data_arrays = block.data_arrays

        data_indeces = np.sort([d["pointIndex"] for d in data["points"]])
        if data_indeces.shape[0] > 300:
            plot_data_indeces = np.sort(
                np.random.choice(len(data["points"]), 300, replace=False)
            )
        else:
            plot_data_indeces = data_indeces

        pulses = data_arrays["pulses"]
        channels = data_arrays["channels"][:]
        current_channel = int(umap_embedding.split("_")[-1])
        selected_channel = current_channel == channels

        try:
            pulse_min = data_arrays["prominent_pulses"][:]
        except KeyError:
            pulse_min = np.ones_like(channels, dtype=np.bool)

        indeces_pulses = np.where((pulse_min & selected_channel))[0]
        colors = px.colors.qualitative.Vivid[: len(umap_selections)]

        fig = go.Figure()
        vertices_current_selection = get_vertices_current_selection(data)
        polygon_current_selection = Polygon(vertices_current_selection)

        log.debug(
            f"Current selection bounds {polygon_current_selection.bounds}"
        )

        for i, (umap_selection_index, umap_selection_data) in enumerate(
            umap_selections.items()
        ):
            vertices_umap_selection = get_vertices_umap_selection(
                umap_selection_data
            )
            polygon_umap_selection = Polygon(vertices_umap_selection)
            log.debug(f"Umap selection Bounds {polygon_umap_selection.bounds}")
            same_bounds = np.allclose(
                np.array(polygon_current_selection.bounds),
                np.array(polygon_umap_selection.bounds),
            )
            log.debug(same_bounds)
            if umap_selection_index in saved_selections:
                if len(figure["layout"]["selections"]) == 1:
                    log.debug("UpdatingSingle Selection")
                    upper, low, mean_wf = calc_mean_wavforms_from_umap(
                        pulses, indeces_pulses[data_indeces]
                    )
                    saved_selections[umap_selection_index]["mean_wf"] = mean_wf
                    saved_selections[umap_selection_index]["upper"] = upper
                    saved_selections[umap_selection_index]["lower"] = low
                    fig = plot_mean_waveforms_from_umap(
                        fig,
                        low,
                        upper,
                        mean_wf,
                        d.metadata.samplerate,
                        pulses[indeces_pulses[plot_data_indeces]],
                        colors[i],
                        colors[i],
                    )
                elif same_bounds:
                    log.debug("Updating reocurring selection")

                    upper, low, mean_wf = calc_mean_wavforms_from_umap(
                        pulses, indeces_pulses[data_indeces]
                    )
                    saved_selections[umap_selection_index]["mean_wf"] = mean_wf
                    saved_selections[umap_selection_index]["upper"] = upper
                    saved_selections[umap_selection_index]["lower"] = low
                    fig = plot_mean_waveforms_from_umap(
                        fig,
                        low,
                        upper,
                        mean_wf,
                        d.metadata.samplerate,
                        pulses[indeces_pulses[plot_data_indeces]],
                        colors[i],
                        colors[i],
                    )
                else:
                    mean_wf = saved_selections[umap_selection_index]["mean_wf"]
                    upper = saved_selections[umap_selection_index]["upper"]
                    low = saved_selections[umap_selection_index]["lower"]
                    log.debug("Taking Saved selection")
                    fig = plot_only_mean_waveforms_from_umap(
                        fig,
                        low,
                        upper,
                        mean_wf,
                        d.metadata.samplerate,
                        colors[i],
                    )
            else:
                saved_selections[umap_selection_index] = umap_selection_data
                upper, low, mean_wf = calc_mean_wavforms_from_umap(
                    pulses, indeces_pulses[data_indeces]
                )
                saved_selections[umap_selection_index]["mean_wf"] = mean_wf
                saved_selections[umap_selection_index]["upper"] = upper
                saved_selections[umap_selection_index]["lower"] = low
                fig = plot_mean_waveforms_from_umap(
                    fig,
                    low,
                    upper,
                    mean_wf,
                    d.metadata.samplerate,
                    pulses[indeces_pulses[plot_data_indeces]],
                    colors[i],
                    colors[i],
                )
                log.debug("creating saved selection")

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event+select",
        )
        nix_file.close()
        return fig, saved_selections

    @app.callback(
        Input("vis_tabs", "active_tab"),
        Input("filepath", "data"),
        Input("select_umap_embedding", "value"),
        Input("num_hdbscan_cluster_size", "value"),
        Input("bt_umap_save_unit", "n_clicks"),
    )
    def save_unit_from_umap(
        tabs, filepath, umap_embedding, cluster_size, bt_saveunit
    ):
        if tabs and tabs != "tab_umap":
            return None
        if not filepath:
            return None
        if not filepath["data_path"]:
            return None

        button = ctx.triggered_id == "bt_umap_save_unit"
        if not button:
            return None

        d = load_data(**filepath)

        save_path = list(Path(d.paths.save_path).rglob("*pulses.*"))

        try:
            save_file = [p for p in save_path if p.suffix in [".nix", ".h5"]][
                0
            ]
        except IndexError:
            return default_umap_figure

        if not save_file.exists:
            return None

        nix_file = nixio.File(str(save_file), nixio.FileMode.ReadWrite)
        block = nix_file.blocks[0]
        data_arrays = block.data_arrays
        embedding = block.data_arrays[f"{umap_embedding}"]
        pulses = block.data_arrays["pulses"]

        if not umap_embedding:
            log.debug("No umap embedding selected, or found")
            return default_umap_figure()

        log.debug(f"Cluster Size {cluster_size}")
        if not cluster_size:
            hdb = HDBSCAN()
        else:
            hdb = HDBSCAN(min_cluster_size=cluster_size)

        hdb_labels = hdb.fit_predict(embedding[:])

        try:
            labels = block.create_data_array(
                "labels",
                "thunderpulse.labels",
                data=np.full(pulses.shape[0], fill_value=-1),
            )
        except DuplicateName:
            labels = block.data_arrays["labels"]

        channels = data_arrays["channels"][:]
        current_channel = int(umap_embedding.split("_")[-1])
        selected_channel = current_channel == channels

        try:
            pulse_min = data_arrays["prominent_pulses"][:]
        except KeyError:
            pulse_min = np.ones_like(channels, dtype=np.bool)

        indeces_pulses = np.where(pulse_min & selected_channel)[0]
        labels[indeces_pulses] = hdb_labels
        log.debug(f"Finished writing labels for {umap_embedding}")
        nix_file.close()
        return None


def calc_mean_wavforms_from_umap(pulses, index):
    mean_wf = np.mean(pulses[index], axis=0)
    std_wf = np.std(pulses[index], axis=0)
    upper = mean_wf + std_wf
    low = mean_wf - std_wf
    return upper, low, mean_wf


def plot_mean_waveforms_from_umap(
    fig, low, upper, mean_wf, sample_rate, mean_pulses, color1, color2
):
    time_slice = np.arange(mean_wf.shape[0]) / sample_rate
    for p in mean_pulses:
        fig.add_trace(
            go.Scattergl(
                x=time_slice,
                y=p,
                mode="lines",
                marker_color=color2,
                showlegend=False,
                opacity=0.2,
            ),
        )
    fig.add_trace(
        go.Scattergl(
            name="upper",
            x=np.concatenate([time_slice, time_slice[::-1]]),
            y=np.concatenate([upper, low[::-1]]),
            mode="lines",
            showlegend=False,
            fill="toself",
            marker_color=color1,
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=time_slice,
            y=mean_wf,
            mode="lines",
            marker_color=color1,
        ),
    )

    return fig


def plot_only_mean_waveforms_from_umap(
    fig, low, upper, mean_wf, sample_rate, color1
):
    time_slice = np.arange(len(mean_wf)) / sample_rate
    fig.add_trace(
        go.Scattergl(
            name="upper",
            x=np.concatenate([time_slice, time_slice[::-1]]),
            y=np.concatenate([upper, low[::-1]]),
            mode="lines",
            showlegend=False,
            fill="toself",
            marker_color=color1,
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=time_slice,
            y=mean_wf,
            mode="lines",
            marker_color=color1,
        ),
    )

    return fig


def parse_lasso_path(path_data):
    """Convert SVG path string to array of polygon vertices"""
    vertices = []
    path = path_data.split("Z")[0]  # Remove closing command
    parts = path.replace("M", "L").split("L")  # Split into segments

    for part in parts:
        if not part.strip():
            continue
        coords = part.strip().split(",")
        if len(coords) >= 2:
            try:
                x = float(coords[0])
                y = float(coords[1])
                vertices.append((x, y))
            except ValueError:
                continue
    return np.array(vertices)


def get_vertices_current_selection(data) -> np.ndarray:
    if "range" in data.keys():
        current_selection = np.array(
            [
                [data["range"]["x"][0], data["range"]["y"][0]],
                [data["range"]["x"][0], data["range"]["y"][1]],
                [data["range"]["x"][1], data["range"]["y"][1]],
                [data["range"]["x"][1], data["range"]["y"][0]],
                [data["range"]["x"][0], data["range"]["y"][0]],
            ]
        )

    else:
        current_selection = np.hstack(
            (
                np.array(data["lassoPoints"]["x"]).reshape(-1, 1),
                np.array(data["lassoPoints"]["y"]).reshape(-1, 1),
            )
        ).reshape(-1, 2)
    return current_selection


def get_vertices_umap_selection(data) -> np.ndarray:
    if data["type"] == "rect":
        vertices = np.array(
            [
                [data["x0"], data["y0"]],
                [data["x0"], data["y1"]],
                [data["x1"], data["y1"]],
                [data["x1"], data["y0"]],
                [data["x0"], data["y0"]],
            ]
        )
    else:
        vertices = parse_lasso_path(data["path"])

    return vertices
