import copy

import nixio
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, ctx
from shapely import Point, Polygon
from sklearn.cluster import HDBSCAN


def default_umap_figure():
    fig = go.Figure(
        data=go.Scattergl(),
    )
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
        Input("num_channel_umap", "value"),
        Input("sw_segments", "value"),
        Input("num_hdb_scan", "value"),
        Input("bt_show_umap_channel", "n_clicks"),
    )
    def create_umap_embeding(
        tabs, filepath, channel, segments, hdb_cluster_size, bt_show_umap
    ):
        if tabs:
            if not tabs == "tab_umap":
                fig = default_umap_figure()
                return fig

        if not channel and channel != 0:
            fig = default_umap_figure()
            return fig

        button = ctx.triggered_id == "bt_show_umap_channel"
        if not button:
            fig = default_umap_figure()
            return fig

        if segments:
            fig = plot_segments(filepath, channel)
        else:
            fig = plot_umap(filepath, channel, hdb_cluster_size)
        return fig

    @app.callback(
        Input("vis_tabs", "active_tab"),
        Input("filepath", "data"),
        Input("num_channel_umap", "value"),
        Input("num_hdb_scan", "value"),
        Input("bt_saveunit", "n_clicks"),
    )
    def save_unit_from_umap(
        tabs, filepath, channel, hdb_cluster_size, bt_saveunit
    ):
        if tabs:
            if not tabs == "tab_umap":
                return

        button = ctx.triggered_id == "bt_saveunit"
        if not button:
            return

        nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadWrite)
        block = nix_file.blocks[0]
        data_arrays = nix_file.blocks[0].data_arrays
        spike_frame = block.data_frames["spike_times_dataframe_processed"]

        try:
            spike_frame.append_column(
                -np.ones(spike_frame.shape[0], dtype=np.int8), "unit"
            )
        except ValueError:
            pass
        try:
            embedding = data_arrays[f"umap_channel_{channel}"]
        except KeyError:
            return

        if hdb_cluster_size:
            hdb = HDBSCAN(min_cluster_size=hdb_cluster_size, n_jobs=-1)
            labels = hdb.fit_predict(embedding[:, :2])
        else:
            labels = embedding[:, 3]
        channel_index = np.where(spike_frame["channel"] == channel)[0]
        spikes_channel = spike_frame.read_rows(
            spike_frame["channel"] == channel
        )
        spikes_channel["unit"][: len(labels)] = labels
        spike_frame.write_rows(
            spikes_channel[: len(labels)], channel_index[: len(labels)]
        )
        print(f"finished {channel}")
        nix_file.close()

    @app.callback(
        Output("umap_selection", "figure"),
        Output("store_umap_selection", "data"),
        Input("filepath", "data"),
        Input("num_channel_umap", "value"),
        Input("umap", "selectedData"),
        Input("waveform_higher", "value"),
        Input("waveform_lower", "value"),
        Input("umap", "figure"),
        Input("store_umap_selection", "data"),
    )
    def plot_selected_umap_embedding(
        filepath, channel, data, higher, lower, figure, saved_selections
    ):
        if not data:
            fig = default_umap_figure()
            return fig, None

        try:
            figure["layout"]["selections"]
        except KeyError:
            fig = default_umap_figure()
            return fig, None

        current_umap_selections = {
            f"{i}": s
            for i, s in enumerate(figure["layout"]["selections"])
            if figure["layout"]["selections"]
        }
        if not saved_selections:
            saved_selections = copy.deepcopy(current_umap_selections)

        colors = px.colors.qualitative.Vivid[: len(current_umap_selections)]
        fig = go.Figure()

        for i, (key, s) in enumerate(current_umap_selections.items()):
            if s["type"] == "rect" and "range" in data.keys():
                vertices = np.array(
                    [
                        [s["x0"], s["y1"]],
                        [s["x0"], s["y0"]],
                        [s["x1"], s["y0"]],
                        [s["x1"], s["y1"]],
                        [s["x0"], s["y1"]],
                    ]
                )
                current_selection = np.array(
                    [
                        [data["range"]["x"][0], data["range"]["y"][0]],
                        [data["range"]["x"][0], data["range"]["y"][1]],
                        [data["range"]["x"][1], data["range"]["y"][1]],
                        [data["range"]["x"][1], data["range"]["y"][0]],
                        [data["range"]["x"][0], data["range"]["y"][0]],
                    ]
                )

            elif s["type"] == "path" and "lassoPoints" in data.keys():
                vertices = parse_lasso_path(s["path"])
                current_selection = np.hstack(
                    (
                        np.array(data["lassoPoints"]["x"]).reshape(-1, 1),
                        np.array(data["lassoPoints"]["y"]).reshape(-1, 1),
                    )
                ).reshape(-1, 2)
                if np.all(current_selection[-1] == vertices[0]):
                    vertices = np.vstack((vertices, vertices[0]))
            else:
                vertices = np.zeros((1, 1))
                current_selection = np.zeros((2, 1))

            if vertices.shape[0] != current_selection.shape[0] and np.all(
                vertices[0] == current_selection[0]
            ):
                time_slice = saved_selections[key]["time_slice"]
                std_upper = saved_selections[key]["std_upper"]
                std_lower = saved_selections[key]["std_lower"]
                mean_wf = saved_selections[key]["mean_wf"]
                fig = plot_mean_waveforms_from_umap(
                    fig,
                    time_slice,
                    std_lower,
                    std_upper,
                    mean_wf,
                    color=colors[i],
                )
            else:
                poly = Polygon(current_selection)
                data_frame = [
                    d
                    for d in data["points"]
                    if poly.contains(Point(d["x"], d["y"]))
                ]

                time_slice, std_upper, std_lower, mean_wf = (
                    calc_mean_wavforms_from_umap(
                        filepath, channel, data_frame, higher, lower
                    )
                )
                if key not in saved_selections.keys():
                    saved_selections[key] = current_umap_selections[key]
                else:
                    del saved_selections[key]
                    saved_selections[key] = current_umap_selections[key]
                saved_selections[key]["time_slice"] = time_slice
                saved_selections[key]["std_upper"] = std_upper
                saved_selections[key]["std_lower"] = std_lower
                saved_selections[key]["mean_wf"] = mean_wf
                fig = plot_mean_waveforms_from_umap(
                    fig,
                    time_slice,
                    std_lower,
                    std_upper,
                    mean_wf,
                    color=colors[i],
                )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            clickmode="event+select",
        )
        return fig, saved_selections


def plot_umap(filepath, channel, hdb_cluster_size):
    if not filepath:
        fig = default_umap_figure()
        return fig
    nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
    data_arrays = nix_file.blocks[0].data_arrays
    try:
        embedding = data_arrays[f"umap_channel_{channel}"]
    except KeyError:
        fig = default_umap_figure()
        return fig

    # subset = np.sort(np.random.choice(np.arange(embedding.shape[0]), 5000, replace=False))
    if hdb_cluster_size:
        hdb = HDBSCAN(min_cluster_size=hdb_cluster_size, n_jobs=-1)
        labels = hdb.fit_predict(embedding[:, :2])
    else:
        labels = embedding[:, 2]
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
                    ticktext=[
                        str(int(label)) for label in np.unique(embedding[:, 2])
                    ],
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


def plot_segments(
    filepath,
    channel,
):
    nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
    spike_frame = nix_file.blocks[0].data_frames[
        "spike_times_dataframe_processed"
    ]
    colors = px.colors.qualitative.Dark24
    cluster = np.unique(
        spike_frame.read_rows(spike_frame["channel"] == channel)["segment"]
    )

    data_arrays = nix_file.blocks[0].data_arrays
    try:
        embedding = data_arrays[f"umap_channel_{channel}"]
    except KeyError:
        fig = default_umap_figure()
        return fig
    labels = spike_frame.read_rows(spike_frame["channel"] == channel)[
        "segment"
    ]
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
                    title=dict(text="Segments", side="right"),
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
    return fig


def calc_mean_wavforms_from_umap(filepath, channel, data, higher, lower):
    nix_file = nixio.File(filepath["data_path"], nixio.FileMode.ReadOnly)
    block = nix_file.blocks[0]
    section = nix_file.sections["recording"]
    sample_rate = float(section["samplerate"][0])
    wf = block.data_arrays[f"waveform_channel_{channel}"]
    index = np.sort([d["pointIndex"] for d in data])

    mean_wf = np.mean(wf[index], axis=0)
    std_wf = np.std(wf[index], axis=0)
    upper = mean_wf + std_wf
    low = mean_wf - std_wf
    time_slice = np.arange(lower / 1000, higher / 1000, 1 / sample_rate)
    return time_slice, upper, low, mean_wf


def plot_mean_waveforms_from_umap(fig, time_slice, low, upper, mean_wf, color):
    fig.add_trace(
        go.Scattergl(
            name="upper",
            x=np.concatenate([time_slice, time_slice[::-1]]),
            y=np.concatenate([upper, low[::-1]]),
            mode="lines",
            showlegend=False,
            fill="toself",
            marker_color=color,
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=time_slice,
            y=mean_wf,
            mode="lines",
            marker_color=color,
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
