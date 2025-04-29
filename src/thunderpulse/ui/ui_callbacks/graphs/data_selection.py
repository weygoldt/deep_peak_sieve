import numpy as np


def select_data(
    nix_data_array,
    time_index,
    time_display,
    samplerate,
):
    data_size = nix_data_array.shape[0]
    t_start = time_index
    t_stop_upper = int(t_start + ((time_display * 0.5) * samplerate))
    t_stop_under = int(t_start - ((time_display * 0.5) * samplerate))
    if t_stop_upper >= data_size:
        diff_to_end = data_size - t_start
        t_stop_upper = t_start + diff_to_end
    if t_stop_under <= 0:
        t_stop_under = t_start

    time_slice = np.arange(t_stop_under, t_stop_upper) / samplerate

    return nix_data_array[t_stop_under:t_stop_upper], time_slice
