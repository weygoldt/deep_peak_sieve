import pathlib

from dash import Dash, Input, Output, ctx


def processing_io_callbacks(app: Dash) -> None:
    @app.callback(
        Output("filepath", "data"),
        Input("datapath", "value"),
        Input("savepath", "value"),
        Input("probepath", "value"),
        Input("bt_load_data", "n_clicks"),
    )
    def io_handling(
        datapath: str | None,
        savepath: str | None,
        probepath: str | None,
        bt_load_data: int,
    ) -> dict | None:
        """Save the user input of the io card to a filepath storage.

        Parameters
        ----------
        datapath : str | None
            Path to data
        savepath : str | None
            Path for saving output
        probepath : str | None
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
            "probe_path": "",
        }
        if not datapath or not savepath or not probepath:
            return save_dict

        data_path = pathlib.Path(datapath)
        save_path = pathlib.Path(savepath)
        probe_path = pathlib.Path(probepath)

        if (
            not data_path.exists()
            or not save_path.exists()
            or not probe_path.exists()
        ):
            return save_dict

        save_dict["save_path"] = str(save_path)
        save_dict["probe_path"] = str(probe_path)
        save_dict["data_path"] = str(data_path)

        return save_dict
