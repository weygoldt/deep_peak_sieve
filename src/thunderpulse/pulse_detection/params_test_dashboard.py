# dash_config_editor_min.py
from __future__ import annotations

import json
from dataclasses import fields, replace
from typing import Any, Dict, List, Tuple
from thunderpulse.pulse_detection.collect_peaks import Params

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html


# --------------------------------------------------------------------------- #
# tiny helper to build labelled inputs
# --------------------------------------------------------------------------- #
def _make_input(label: str, value: Any, path: str) -> html.Div:
    cid = {"type": "cfg-input", "path": path}

    if isinstance(value, bool):
        widget = dbc.Checkbox(id=cid, value=value)
    elif isinstance(value, (int, float)):
        widget = dbc.Input(id=cid, type="number", value=value)
    else:
        widget = dbc.Input(id=cid, type="text", value=str(value))

    return dbc.Row(
        [
            dbc.Col(html.Label(label, className="fw-bold"), width=4),
            dbc.Col(widget, width=8),
        ],
        className="mb-2",
    )


def _walk_dc(dc_obj, base="") -> Tuple[List[html.Div], Dict[str, Any]]:
    """Return (components, flat_value_dict)."""
    comps, flat = [], {}
    for f in fields(dc_obj):
        path = f"{base}.{f.name}" if base else f.name
        val = getattr(dc_obj, f.name)
        if hasattr(val, "__dataclass_fields__"):
            inner_comps, inner_flat = _walk_dc(val, path)
            comps.append(
                dbc.Card(
                    dbc.CardBody([html.H6(f.name), *inner_comps]),
                    className="mb-3",
                )
            )
            flat.update(inner_flat)
        else:
            comps.append(_make_input(f.name, val, path))
            flat[path] = val
    return comps, flat


def launch_config_dashboard(params_cls, title="Config editor", port=8050):
    """Minimal Dash GUI – NO file download, JSON preview only."""
    cfg = params_cls()
    components, flat_init = _walk_dc(cfg)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    json_view = dbc.Textarea(
        id="cfg-json", style={"height": "250px"}, readOnly=True
    )

    app.layout = dbc.Container(
        [html.H3(title, className="my-3"), *components, json_view],
        fluid=True,
    )

    # build callback with one Input per control -------------------------------
    inputs = [
        Input({"type": "cfg-input", "path": p}, "value")
        for p in flat_init.keys()
    ]
    states = inputs  # identical, but we could keep Inputs only

    @app.callback(
        Output("cfg-json", "value"), inputs, prevent_initial_call=False
    )
    def _update_json(*vals):
        flat = dict(zip(flat_init.keys(), vals))
        new_cfg = _apply(cfg, flat)
        return json.dumps(new_cfg.to_dict(), indent=2)

    app.run(debug=False, port=port)


def _apply(root, flat):
    """Return new dataclass tree updated with values from flat dict."""

    def rec(obj, pref):
        updates = {}
        for f in fields(obj):
            path = f"{pref}.{f.name}" if pref else f.name
            if any(k == path or k.startswith(path + ".") for k in flat):
                val = getattr(obj, f.name)
                if hasattr(val, "__dataclass_fields__"):
                    updates[f.name] = rec(val, path)
                elif path in flat:
                    updates[f.name] = type(val)(flat[path])
        return replace(obj, **updates)

    return rec(root, "")


if __name__ == "__main__":
    launch_config_dashboard(Params, title="Peak-detection parameters")
