from scipy.signal import butter, sosfiltfilt


def bandpass_filter(data, lowcut, highcut, sample_rate, order=2):
    sos = butter(order, [lowcut, highcut], btype="band", fs=sample_rate, output="sos")
    y = sosfiltfilt(sos, data)
    return y
