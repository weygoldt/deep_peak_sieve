import dash_bootstrap_components as dbc
from dash import dash_table


def create_peak_table_card():
    peak_table_card = dbc.Card(
        dbc.CardBody(
            [
                dash_table.DataTable(
                    id="peaks_table",
                    page_current=0,
                    page_size=20,
                    page_action="custom",
                    style_header={
                        "backgroundColor": "rgb(30, 30, 30)",
                        "color": "white",
                    },
                    style_data={
                        "backgroundColor": "rgb(50, 50, 50)",
                        "color": "white",
                    },
                    style_as_list_view=True,
                    sort_action="custom",
                    sort_mode="single",
                    sort_by=[],
                ),
            ]
        )
    )
    return peak_table_card
