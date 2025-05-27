import dash_bootstrap_components as dbc
from dash import html


def create_paths_offcanvas() -> dbc.Card:
    """Create and return an IO Card.

    Returns
    -------
    dbc.Card

    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.P("Data Folder", className="card-text"),
                dbc.Textarea(
                    # type="text",
                    placeholder="Please put in your datapath",
                    id="datapath",
                    persistence=True,
                    persistence_type="local",
                    className="card-text",
                ),
                html.Br(),
                html.P("Save Path", className="card-text"),
                dbc.Textarea(
                    # type="text",
                    placeholder="Please put in your save path",
                    id="savepath",
                    persistence=True,
                    persistence_type="local",
                    className="card-text",
                ),
                html.Br(),
                html.P("Layout of Probe/Grid", className="card-text"),
                dbc.Textarea(
                    placeholder="Please put in your path to the probe/grid layout",
                    id="probepath",
                    persistence=True,
                    persistence_type="local",
                    className="card-text",
                ),
                html.Br(),
                dbc.Checklist(
                    options=[
                        {
                            "label": "Run for all parent directories",
                            "value": 0,
                        }
                    ],
                    id="sw_paths_run_all_parent_dirs",
                    switch=True,
                    persistence=True,
                    persistence_type="local",
                ),
                html.Br(),
            ]
        )
    )
