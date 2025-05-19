import numpy as np
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output
from IPython import embed
from IPython.core.interactiveshell import is_integer_string
from plotly import subplots

from thunderpulse.data_handling.data import load_data
from thunderpulse.pulse_detection.config import (
    BandpassParameters,
    FiltersParameters,
    FindPeaksKwargs,
    NotchParameters,
    Params,
    PeakDetectionParameters,
    PostProcessingParameters,
    PreProcessingParameters,
    SavgolParameters,
)
from thunderpulse.pulse_detection.detection import (
    apply_filters,
    detect_peaks_on_block,
)
from thunderpulse.ui_callbacks.graphs import data_selection as ds
from thunderpulse.ui_callbacks.graphs.channel_selection import select_channels
from thunderpulse.utils.check_config import check_config_params
from thunderpulse.utils.loggers import get_logger
from thunderpulse.utils.logging_setup import setup_logging

log = get_logger(__name__)
setup_logging(log)


def callbacks(app):
    @app.callback(
        Output("pulse_detection_config", "data"),
        # Filter
        inputs={
            "general": {
                "filepath": Input("filepath", "data"),
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
            "general_pulse": {
                "buffersize_s": Input("num_pulse_buffersize", "value")
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
                "distance_channels": Input("num_pulse_distance", "value"),
            },
            "findpeaks": {
                "height": Input("num_findpeaks_height", "value"),
                "threshold": Input("num_findpeaks_threshold", "value"),
                "distance": Input("num_findpeaks_distance", "value"),
                "prominence": Input("num_findpeaks_prominence", "value"),
                "width": Input("num_findpeaks_width", "value"),
            },
            "postprocessing": {
                "enable_resampling": Input("sw_resampling_enable", "value"),
                "n_resamples": Input("num_resampling_n", "value"),
                "enable_centering": Input("sw_sample_centering", "value"),
                "enable_sign_correction": Input(
                    "sw_sample_sign_correction", "value"
                ),
                "centering_method": Input(
                    "select_sample_centering_method", "value"
                ),
                "polarity": Input(
                    "select_sample_sign_correction_polarity", "value"
                ),
            },
        },
    )
    def update_config_params(
        general: dict,
        pre_filter: dict,
        savgol: dict,
        bandpass: dict,
        notch: dict,
        general_pulse: dict,
        pulse: dict,
        findpeaks: dict,
        postprocessing,
    ):
        filepath = general["filepath"]
        if not filepath:
            return None
        if not filepath["data_path"]:
            return None

        d = load_data(**filepath)

        log.debug("Updating pulse detection config")

        # NOTE: rewrite checkbox input [1]/True []/False to bool
        pre_filter["common_median_reference"] = bool(
            pre_filter["common_median_reference"]
        )
        prefilter = PreProcessingParameters(**pre_filter)

        # apply_filters_names = []
        # apply_filters_params = []
        # filter_params_function = [savgol, bandpass, notch]
        # filter_names = FiltersParameters().filters
        # filter_params = [SavgolParameters, BandpassParameters, NotchParameters]
        # for f_name, f_params, f_params_func in zip(
        #     filter_names, filter_params, filter_params_function, strict=True
        # ):
        #     check_f = check_config_params(f_params_func)
        #     if check_f:
        #         apply_filters_params.append(f_params(**f_params_func))
        #         apply_filters_names.append(f_name)
        # savgol_filter = SavgolParameters(**savgol)
        # bandpass_filter = BandpassParameters(**bandpass)
        # notch_filter = NotchParameters(**notch)
        #
        filters = FiltersParameters(
            savgol=SavgolParameters(**savgol),
            bandpass=BandpassParameters(**bandpass),
            notch=NotchParameters(**notch),
        )
        filters.remove_filters_with_all_none_params()
        # filters = FiltersParameters(
        #     filters=apply_filters_names, filter_params=apply_filters_params
        # )

        findpeaks = FindPeaksKwargs(**findpeaks)
        peaks = PeakDetectionParameters(**pulse, find_peaks_kwargs=findpeaks)

        postprocessing["enable_resampling"] = bool(
            postprocessing["enable_resampling"]
        )
        postprocessing["enable_sign_correction"] = bool(
            postprocessing["enable_sign_correction"]
        )
        postprocessing["enable_centering"] = bool(
            postprocessing["enable_centering"]
        )
        postpros = PostProcessingParameters(**postprocessing)
        params = Params(
            prefilter,
            filters,
            peaks,
            postpros,
            sensoryarray=d.sensorarray,
            **general_pulse,
        )
        return params.to_dict()
