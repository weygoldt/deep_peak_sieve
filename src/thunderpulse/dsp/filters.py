"""Signal processing filters"""

from scipy.signal import butter, filtfilt, iirnotch, savgol_filter, sosfiltfilt

from thunderpulse.utils.loggers import get_logger

log = get_logger(__name__)


def savitzky_golay_filter(data, fs, window_length_s, polyorder):
    window_length = int(window_length_s * fs)
    log.debug(f"Window length: {window_length} samples")
    if window_length % 2 == 0:
        window_length += 1  # Make sure the window length is odd
        log.debug(
            f"Window length was even, increased to {window_length} samples"
        )
    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder,
        axis=-1,
    )


def bandpass_filter(data, fs, lowcut, highcut, order):
    coeffs = butter(
        order,
        [lowcut, highcut],
        btype="band",
        analog=False,
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos=coeffs, x=data, axis=-1)


def notch_filter(data, fs, notch, q, order):
    b, a = iirnotch(notch, q, fs)
    return filtfilt(b, a, data, axis=-1)
