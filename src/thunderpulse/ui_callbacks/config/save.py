import json
import pathlib

import nixio
from dash import Dash, Input, Output, ctx
from IPython import embed

from thunderpulse.data_handling.data import load_data
from thunderpulse.pulse_detection.config import (
    BandpassParameters,
    FiltersParameters,
    FindPeaksKwargs,
    NotchParameters,
    Params,
    PeakDetectionParameters,
    PrefilterParameters,
    ResampleParameters,
    SavgolParameters,
)
from thunderpulse.utils.check_config import check_config_params


def callbacks(app: Dash) -> None:
    @app.callback(
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
                "save_button": Input("bt_save_config", "n_clicks"),
            },
            "pre_filter": {
                "common_median_reference": Input(
                    "sw_common_reference", "value"
                ),
            },
            "savgol": {
                "window_length_s": Input("num_savgol_window_length", "value"),
                "polyorder": Input("num_savgol_polyorder", "value"),
            },
            "bandpass": {
                "lowcut": Input("num_bandpass_lowcutoff", "value"),
                "highcut": Input("num_bandpass_highcutoff", "value"),
            },
            "notch": {
                "notch_freq": Input("num_notchfilter_freq", "value"),
                "quality_factor": Input("num_notchfilter_quality", "value"),
            },
            "pulse": {
                "min_channels": Input("num_pulse_min_channels", "value"),
                "mode": Input("select_pulse_mode", "value"),
                "min_peak_distance_s": Input(
                    "num_pulse_min_peak_distance", "value"
                ),
                "cutout_window_around_peak_s": Input(
                    "num_pulse_waveform", "value"
                ),
            },
            "findpeaks": {
                "height": Input("num_findpeaks_height", "value"),
                "threshold": Input("num_findpeaks_threshold", "value"),
                "distance": Input("num_findpeaks_distance", "value"),
                "prominence": Input("num_findpeaks_prominence", "value"),
                "width": Input("num_findpeaks_width", "value"),
            },
            "resample": {
                "enabled": Input("sw_resampling_enable", "value"),
                "n_resamples": Input("num_resampling_n", "value"),
            },
        },
        prevent_initial_call=True,
    )
    def save_config(
        general: dict,
        pre_filter,
        savgol,
        bandpass,
        notch,
        pulse,
        findpeaks,
        resample,
    ) -> dict | None:
        button = ctx.triggered_id == "bt_save_config"
        if not button:
            return None
        filepath, save_button = general.values()
        if not filepath:
            return None
        if not filepath["save_path"]:
            return None
        d = load_data(**filepath)

        apply_filters_names = []
        apply_filters_params = []
        filter_params_function = [savgol, bandpass, notch]
        filter_names = FiltersParameters().filters
        filter_params = [SavgolParameters, BandpassParameters, NotchParameters]
        for f_name, f_params, f_params_func in zip(
            filter_names, filter_params, filter_params_function, strict=True
        ):
            check_f = check_config_params(f_params_func)
            if check_f:
                apply_filters_params.append(f_params(**f_params_func))
                apply_filters_names.append(f_name)

        prefilter = PrefilterParameters(**pre_filter)

        filters = FiltersParameters(
            filters=apply_filters_names, filter_params=apply_filters_params
        )

        findpeaks = FindPeaksKwargs(**findpeaks)
        peaks = PeakDetectionParameters(**pulse, find_peaks_kwargs=findpeaks)
        resample = ResampleParameters(**resample)
        params = Params(prefilter, filters, peaks, resample)
        current_parms = params.to_dict()

        save_path = pathlib.Path(filepath["save_path"]) / "config.nix"

        file = nixio.File(str(save_path), nixio.FileMode.Overwrite)
        sec = file.create_section("general", "thunderpulse.general")
        # TODO: Fix metatdata cration if it contains None
        # create_metadata_from_dict(current_parms, sec)
        file.close()

        with open(save_path.with_suffix(".json"), "w") as f:
            f.write(params.to_json())


def create_metadata_from_dict(d, section: nixio.Section):
    for key, value in d.items():
        if isinstance(value, dict):
            new_sec = section.create_section(key, f"{type(key)}")
            create_metadata_from_dict(value, new_sec)
        else:
            try:
                section.create_property(key, values_or_dtype=value)
            except TypeError:
                if isinstance(value, list):
                    value = [str(i) for i in value]
                else:
                    value = str(value)
                section.create_property(key, values_or_dtype=value)
