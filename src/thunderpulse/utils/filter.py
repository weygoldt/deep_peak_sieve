from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    coeffs = butter(
        order,
        [lowcut, highcut],
        btype="band",
        analog=False,
        fs=fs,
        output="sos",
    )

    y = sosfiltfilt(sos=coeffs, x=data, axis=0)
    return y


def notch_filter(data, notch, fs, Q=40, order=5):
    # coeffs = butter(
    #     order,
    #     [notch-2, notch+2],
    #     btype="bandstop",
    #     analog=False,
    #     fs=fs,
    #     output="sos",
    # )
    # y = sosfiltfilt(sos=coeffs, x=data, axis=0)
    b, a = iirnotch(notch, Q, fs)
    y = filtfilt(b, a, data, axis=0)
    return y
