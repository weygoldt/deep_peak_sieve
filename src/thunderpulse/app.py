from typing import Annotated

import dash_bootstrap_components as dbc
import typer
from dash import Dash, dcc, html

from thunderpulse import ui_callbacks, ui_layout
from thunderpulse.utils.loggers import configure_logging, get_logger

typer_app = typer.Typer()
log = get_logger(__name__)


@typer_app.command()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="Increase verbosity."
        ),
    ] = 0,
) -> None:
    """Generate the Dash app."""
    configure_logging(verbose)
    log.info("Starting Thunderpulse Dashboard")

    app = Dash(
        external_stylesheets=[dbc.themes.DARKLY],
    )

    header = html.H4(children="Thunderpulse", style={"textAlign": "center"})
    channel_slider = ui_layout.channel_slider.create_channel_slider()
    time_slider = ui_layout.time_slider.create_time_slider()
    layout_graph_probe = ui_layout.probe_graph.create_layout_probe_graph()
    visualization_tabs = ui_layout.graphs.create_visualization_tabs()
    config_tabs = ui_layout.config.combine.create_config()
    peak_table = ui_layout.peak_table.create_peak_table_card()

    app.layout = dbc.Container(
        [
            header,
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            config_tabs,
                            peak_table,
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
                                    # dcc.Store(id="store_umap_selection"),
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

    log.info("Initializing callbacks")
    log.info("Initializing config tabs")
    ui_callbacks.config.detection_card.callbacks(app)
    ui_callbacks.config.filter_card.callbacks(app)
    ui_callbacks.config.path_card.callbacks(app)
    ui_callbacks.config.offcanvas.callbacks(app)
    ui_callbacks.config.load.callbacks(app)
    ui_callbacks.config.save.callbacks(app)

    ui_callbacks.channel_slider.callbacks(app)
    ui_callbacks.time_slider.callbacks(app)

    log.info("Initializing graphs")
    ui_callbacks.graphs.traces.callbacks_traces(app)
    ui_callbacks.graphs.probe.callbacks_sensory_array(app)
    ui_callbacks.keyboard_shortcuts.create_shortcuts(app)

    ui_callbacks.peak_table.callbacks(app)

    app.run(debug=True)


if __name__ == "__main__":
    typer_app()
