import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

# import thunderpulse.tables as tables
# from thunderpulse import graphs, processing, ui
from thunderpulse import ui, ui_callbacks


def main() -> None:
    """Generate the Dash app."""
    app = Dash(
        external_stylesheets=[dbc.themes.DARKLY],
    )

    header = html.H3(children="Thunderpulse", style={"textAlign": "center"})
    channel_slider = ui.channel_slider.create_channel_slider()
    time_slider = ui.time_slider.create_time_slider()
    layout_graph_probe = ui.probe_graph.create_layout_probe_graph()
    visualization_tabs = ui.visualization_tabs.create_visualization_tabs()
    config_tabs = ui.config_tabs.combine_config_cards.create_config_tabs()

    app.layout = dbc.Container(
        [
            header,
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            config_tabs,
                            layout_graph_probe,
                            dbc.Spinner(
                                dcc.Store(id="filepath"),
                                color="info",
                                fullscreen=True,
                                fullscreen_style={
                                    "background-color": "transparent",
                                    # "border": "none",  # Optional: Remove the border as well
                                },
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            channel_slider,
                            html.Div(
                                [
                                    html.H5(
                                        children="Traces",
                                        style={"textAlign": "center"},
                                    ),
                                    visualization_tabs,
                                    dcc.Store(id="store_umap_selection"),
                                    time_slider,
                                    dcc.Store(id="peak_storage"),
                                    html.Br(),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        # extents if free space is available
        fluid=True,
    )

    ui_callbacks.config_tabs.preprocessing_card.callbacks(app)
    ui_callbacks.config_tabs.io_card.callbacks(app)
    ui_callbacks.channel_slider.callbacks(app)
    ui_callbacks.time_slider.callbacks(app)

    # ui.callbacks..processing_io_callbacks(app)
    # processing.save_processing.callback_save_processing_channels(app)
    # processing.waveforms.callbacks_create_waveforms(app)
    # processing.calc_umap.callbacks_create_umap_embedding(app)

    ui_callbacks.graphs.traces.callbacks_traces(app)
    ui_callbacks.graphs.probe.callbacks_probe(app)

    # NOTE: NOT WORKING
    # ui_callbacks.graphs.psd.callbacks_psd(app)
    # ui_callbacks.graphs.spikes_ampl.callbacks_spikes_ampl(app)
    # ui_callbacks.graphs.waveforms.callback_waveforms(app)
    # ui_callbacks.graphs.dashumap.callbacks_umap(app)

    # tables.peak_table_window.callbacks_peak_table_window(app)

    app.run(debug=True)


if __name__ == "__main__":
    main()
