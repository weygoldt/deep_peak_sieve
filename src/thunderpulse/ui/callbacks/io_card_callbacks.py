import pathlib
from dataclasses import asdict

from dash import Dash, Input, Output, ctx
from IPython import embed

from thunderpulse.data_handling.data import load_data


def callbacks_io(app: Dash) -> None:
    @app.callback(
        Output("datapath", "invalid"),
        Output("datapath", "valid"),
        Input("datapath", "value"),
    )
    def datapath_feedback(datapath):
        if not datapath:
            return True, False
        datapath = pathlib.Path(datapath)
        if datapath.exists():
            return False, True
        return True, False

    @app.callback(
        Output("savepath", "invalid"),
        Output("savepath", "valid"),
        Input("savepath", "value"),
    )
    def savepath_feedback(savepath):
        if not savepath:
            return True, False
        savepath = pathlib.Path(savepath)
        if savepath.exists():
            return False, True
        return True, False

    @app.callback(
        Output("probepath", "invalid"),
        Output("probepath", "valid"),
        Input("probepath", "value"),
    )
    def probepath_feedback(probepath):
        if not probepath:
            return True, False
        probepath = pathlib.Path(probepath)
        if probepath.exists() and probepath.is_file():
            return False, True
        return True, False

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
