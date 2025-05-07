import numpy as np
import polars as pl
from dash import Input, Output
from IPython import embed


def callbacks(app):
    @app.callback(
        Output("peaks_table", "data"),
        Output("peaks_table", "columns"),
        Output("peaks_table", "page_count"),
        Input("peaks_table", "page_current"),
        Input("peaks_table", "page_size"),
        Input("peaks_table", "sort_by"),
        Input("peak_storage", "data"),
    )
    def update_peak_table(page_current, page_size, sort_by, peaks_storage):
        if not peaks_storage:
            return None, None, None

        peaks_storage.pop("pulses")
        peaks_storage.pop("channels")
        peaks_storage.pop("all_channels")
        peaks_storage.pop("all_pulses")

        starts = np.array(peaks_storage["start_stop_index"])[:, 0]
        stops = np.array(peaks_storage["start_stop_index"])[:, 1]

        peaks_storage["start"] = starts
        peaks_storage["stop"] = stops
        peaks_storage.pop("start_stop_index")

        dataframe = pl.DataFrame(peaks_storage)
        data_frame_size = dataframe.shape[0]

        if len(sort_by):
            dataframe = dataframe.sort(
                pl.col(sort_by[0]["column_id"]),
                descending=sort_by[0]["direction"] != "asc",
            )

        columns = [{"name": i, "id": i} for i in dataframe.columns]

        dataframe_ = dataframe.slice(page_current * page_size, page_size)

        return (
            dataframe_.to_dicts(),
            columns,
            int(data_frame_size / page_size),
        )
