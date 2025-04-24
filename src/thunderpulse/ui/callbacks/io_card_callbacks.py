import pathlib
from dash import Output, Input
from probeinterface import read_probeinterface

from IPython import embed


def callbacks_io(app):
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
        else:
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
        else:
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
            try:
                probe = read_probeinterface(probepath)
            except ValueError:
                return True, False
            return False, True
        else:
            return True, False
