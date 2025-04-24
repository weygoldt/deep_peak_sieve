import dash_bootstrap_components as dbc
from .config_tabs.io_card import create_io_card
from .config_tabs.preprocessing_card import create_preprocessing_card
from .config_tabs.peak_table_card import create_peak_table_card
from .config_tabs.cluster_card import create_cluster_card
# from .config_tabs.autoencoder_card import creat


def create_config_tabs():
    io_card = create_io_card()
    preprocessing_card = create_preprocessing_card()
    peak_table_card = create_peak_table_card()
    cluster_card = create_cluster_card()
    # autoencoder_card = create_autoencoder_card()

    tabs = dbc.Card(
        dbc.Tabs(
            [
                dbc.Tab(io_card, label="IO", tab_id="io"),
                dbc.Tab(
                    preprocessing_card, label="Preprocessing", tab_id="preprocessing"
                ),
                dbc.Tab(peak_table_card, label="Peaks", tab_id="peaks"),
                dbc.Tab(cluster_card, label="Clustering", tab_id="clustering"),
                # dbc.Tab(autoencoder_card, label="Autoencoder", tab_id="autoencoder"),
            ],
            id="tabs",
            active_tab="io",
            style={"widht": "20%"},
        ),
        id="styled-numeric-input",
    )
    return tabs
