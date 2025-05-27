from pathlib import Path

from dash import Dash, Input, Output, ctx


def callbacks(app: Dash) -> None:
    @app.callback(
        Output("filepath", "data"),
        Input("datapath", "value"),
        Input("savepath", "value"),
        Input("probepath", "value"),
        Input("bt_load_data", "n_clicks"),
        prevent_initial_call=True,
    )
    def io_handling(
        datapath: str | None,
        savepath: str | None,
        sensorarraypath: str | None,
        bt_load_data: int,
    ) -> dict | None:
        """Save the user input of the io card to a filepath storage.

        Parameters
        ----------
        datapath : str | None
            Path to data
        savepath : str | None
            Path for saving output
        sensorarray_path: str | None
            Path to Probelayout / Gridlayout
        bt_load_data : int
            Button load data is needed for ctx.triggered_id check

        Returns
        -------
        dict | None
            Filepath Storage that dash is saveing in the Browser

        """
        button = ctx.triggered_id == "bt_load_data"

        if not button:
            return None

        save_dict = {
            "data_path": "",
            "save_path": "",
            "sensorarray_path": "",
        }
        if not datapath or not savepath or not sensorarraypath:
            return save_dict

        data_path = Path(datapath)
        save_path = Path(savepath)
        sensorarray_path = Path(sensorarraypath)

        if (
            not data_path.exists()
            or not save_path.exists()
            or not sensorarray_path.exists()
        ):
            return save_dict

        save_dict["save_path"] = str(save_path)
        save_dict["sensorarray_path"] = str(sensorarray_path)
        save_dict["data_path"] = str(data_path)

        return save_dict
