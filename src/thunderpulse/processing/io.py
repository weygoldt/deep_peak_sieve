import pathlib
from dash import Output, Input, ctx
from dash.exceptions import PreventUpdate

from IPython import embed
from .to_nix import write_nix_file


def process_io(datapath, savepath, probepath, bt_overwrite):
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

    if not data_path.exists() or not save_path.exists() or not probe_path.exists():
        return save_dict

    save_dict["save_path"] = str(save_path)
    save_dict["probe_path"] = str(probe_path)
    if not data_path.is_file() and not data_path.suffix == ".nix":
        nix_path = write_nix_file(data_path, save_path, probe_path, bt_overwrite)
        if not nix_path:
            return save_dict
        else:
            save_dict["data_path"] = str(nix_path)
    else:
        save_dict["data_path"] = str(data_path)

    return save_dict


def processing_io_callbacks(app):
    @app.callback(
        Output("filepath", "data"),
        Input("datapath", "value"),
        Input("savepath", "value"),
        Input("probepath", "value"),
        Input("bt_overwrite_nix_file", "n_clicks"),
        Input("bt_load_data", "n_clicks"),
    )
    def io_handling(datapath, savepath, probepath, bt_ovwerwrite, bt_load_data):
        button = ctx.triggered_id == "bt_load_data"
        overwrite = ctx.triggered_id == "bt_overwrite_nix_file"
        save_dict = None
        if button:
            save_dict = process_io(datapath, savepath, probepath, False)
        elif overwrite:
            save_dict = process_io(datapath, savepath, probepath, True)
        else:
            PreventUpdate()
        return save_dict
