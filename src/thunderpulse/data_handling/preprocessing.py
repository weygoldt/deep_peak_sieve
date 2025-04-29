import numpy as np
from IPython import embed
from joblib import Parallel, delayed

from thunderpulse.data_handling import filter


def preprocessing_current_slice(
    sliced_recording,
    sample_rate,
    sw_bandpass=False,
    low=250,
    high=500,
    sw_common_ref=False,
    sw_notch_filter=False,
    notch=500,
):
    if sw_common_ref:
        sliced_recording = sliced_recording - np.median(
            sliced_recording,
            axis=1,
            keepdims=True,
        )
    if sw_bandpass:
        sliced_recording = filter.bandpass_filter(
            sliced_recording, low, high, sample_rate
        )

    if sw_notch_filter:
        sliced_recording = filter.notch_filter(
            sliced_recording, notch, sample_rate
        )

    return sliced_recording


def preprocessing_current_slice_save_to_disk(
    sliced_recording,
    sample_rate,
    sw_bandpass,
    low,
    high,
    sw_common_ref,
    common_ref,
    sw_notch_filter=False,
    notch=500,
):
    if sw_common_ref:
        sliced_recording = sliced_recording - common_ref.flatten()
    if sw_bandpass:
        sliced_recording = filter.bandpass_filter(
            sliced_recording, low, high, sample_rate
        )
    if sw_notch_filter:
        sliced_recording = filter.notch_filter(
            sliced_recording, notch, sample_rate
        )

    return sliced_recording


def parallel_common_ref(sliced_recording):
    return np.median(sliced_recording, axis=1, keepdims=True)


def common_ref_recording(recording, chunks):
    print("Calculate Median for Chunks")
    res = Parallel(n_jobs=-1)(
        delayed(parallel_common_ref)(recording[start:stop])
        for start, stop in zip(chunks[:-1], chunks[1:], strict=False)
    )

    last_chunk = recording.shape[0] - chunks[-1]
    last_median = np.median(recording[-last_chunk:], axis=1, keepdims=True)
    res.append(last_median)
    embed()

    return np.median(np.array(res), axis=0), last_median


def common_ref_recording_channels(recording):
    return np.median(recording, axis=1, keepdims=True)
