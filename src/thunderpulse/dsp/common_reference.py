import numpy as np


def common_median_reference(data):
    return data - np.median(data, axis=1, keepdims=True)


def get_common_median_reference(data):
    return np.median(data, axis=1, keepdims=True)


def common_mean_reference(data):
    return data - np.mean(data, axis=1, keepdims=True)
