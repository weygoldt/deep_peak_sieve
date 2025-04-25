import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

import thunderpulse.graphs as graphs
import thunderpulse.processing as processing
# import thunderpulse.tables as tables
import thunderpulse.ui as ui


def main():
    app = Dash(
        external_stylesheets=[dbc.themes.DARKLY],
    )

    header = html.H3(children="Thunderpulse", style={"textAlign": "center"})
    channel_slider = ui.channel_slider.create_channel_slider()
    time_slider = ui.time_slider.create_time_slider()
    layout_graph_probe = ui.layout_probe_graph.create_layout_probe_graph()
    visualization_tabs = ui.layout_visualization_tabs.create_visualization_tabs()
    config_tabs = ui.layout_config_tabs.create_config_tabs()

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

    ui.callbacks.preprocessing_card_callbacks.callbacks(app)
    ui.callbacks.io_card_callbacks.callbacks_io(app)
    ui.channel_slider.callback_channel_slider(app)
    ui.time_slider.callback_time_slider(app)

    # ui.callbacks..processing_io_callbacks(app)
    processing.save_processing.callback_save_processing_channels(app)
    processing.waveforms.callbacks_create_waveforms(app)
    processing.calc_umap.callbacks_create_umap_embedding(app)

    graphs.traces.callbacks_traces(app)
    graphs.psd.callbacks_psd(app)
    graphs.probe.callbacks_probe(app)
    graphs.spikes_ampl.callbacks_spikes_ampl(app)
    graphs.waveforms.callback_waveforms(app)
    graphs.dashumap.callbacks_umap(app)

    # tables.peak_table_window.callbacks_peak_table_window(app)

    app.run(debug=True)


if __name__ == "__main__":
    main()
