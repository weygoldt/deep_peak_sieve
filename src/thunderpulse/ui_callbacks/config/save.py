import pathlib
from ast import Num
from numbers import Number

import nixio
import numpy as np
from dash import Dash, Input, ctx
from IPython import embed

from thunderpulse.data_handling.data import load_data
from thunderpulse.pulse_detection.config import Params
from thunderpulse.utils.loggers import get_logger
from thunderpulse.utils.logging_setup import setup_logging

log = get_logger(__name__)
setup_logging(log)


def callbacks(app: Dash) -> None:
    @app.callback(
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
                "save_button": Input("bt_save_config", "n_clicks"),
            },
            "pulse_detection_config": Input("pulse_detection_config", "data"),
        },
        prevent_initial_call=True,
    )
    def save_config(
        general: dict,
        pulse_detection_config: dict,
    ) -> dict | None:
        button = ctx.triggered_id == "bt_save_config"
        if not button:
            return None
        filepath, save_button = general.values()
        if not filepath:
            return None
        if not filepath["save_path"]:
            return None
        log.debug("Saving pulse detection parameters")
        d = load_data(**filepath)
        params = Params.from_dict(pulse_detection_config)

        current_parms = params.to_dict()

        save_path = pathlib.Path(filepath["save_path"]) / "config.nix"

        file = nixio.File(str(save_path), nixio.FileMode.Overwrite)
        sec = file.create_section(
            "pulse_detection_config", "thunderpulse.pulse_detection.config"
        )
        create_metadata_from_dict(current_parms, sec)
        embed()

        file.close()

        with open(save_path.with_suffix(".json"), "wb") as f:
            f.write(params.to_json())


def create_metadata_from_dict(d, section: nixio.Section):
    for key, value in d.items():
        if isinstance(value, dict):
            new_sec = section.create_section(key, f"{type(key)}")
            create_metadata_from_dict(value, new_sec)
        else:
            try:
                prop = section.create_property(key, values_or_dtype=value)
                prop.data_type = type(value)
            except TypeError:
                if isinstance(value, list):
                    v = [str(i) for i in value]
                else:
                    v = str(value)
                section.create_property(key, values_or_dtype=v)


def nix_metadata_to_dict(section):
    info = {}
    for p in section.props:
        info[p.name] = (
            [v for v in p.values],
            p.unit if p.unit is not None else "",
        )
    for s in section.sections:
        info[s.name] = nix_metadata_to_dict(s)
    return info


def create_dict_from_section(section: nixio.Section) -> dict:
    d = {}
    for key, value in section.items():
        if isinstance(value, nixio.Section):
            subdict = create_dict_from_section(value)
            d[key] = subdict
        else:
            d[key] = value.values
    return d


def clean_value(val):
    # Convert numpy scalars to Python scalars
    if isinstance(val, (np.generic,)):
        return val.item()
    # Convert string 'None' to None
    if val == "None":
        return None
    # Convert tuple of length 1 to the value
    if isinstance(val, tuple):
        if len(val) == 1:
            return clean_value(val[0])
        else:
            # Convert tuple to list of cleaned values
            return [clean_value(v) for v in val]
    # Recursively clean dicts
    if isinstance(val, dict):
        return {k: clean_value(v) for k, v in val.items()}
    # Recursively clean lists
    if isinstance(val, list):
        return [clean_value(v) for v in val]
    return val


def clean_dict(d):
    return {k: clean_value(v) for k, v in d.items()}
