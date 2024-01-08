#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.stats import zscore


def chdir(dir_, logger, create=False):
    if not dir_.is_dir():
        if create:
            logger.warning(f"The directory {dir_} doesn't exist. Creating it")
            dir_.mkdir(parents=True, exist_ok=True)
        else:
            logger.error(f"The directory {dir_} doesn't exist")
            raise SystemExit(1)


def load_data(indir, channels, fs):

    dset = []
    norm = []
    for channel in channels:

        fname = f'{indir}/{channel}'
        tseries = TimeSeries.read(fname)

        # remove sides due to whitening artifacts
        data = tseries[fs*4:-fs*4].reshape(-1, 1)

        # normalize and record the normalisation factor
        norm_factor = np.max(np.abs(data))
        norm.append(norm_factor)
        dset.append(data / norm_factor)

    dset = np.squeeze(dset).astype('float32')

    return dset, norm


def find_noisy(data, threshold, padding):
    """ Find idx passing the Z-score threshold."""

    noise_idx = np.argwhere(np.abs(zscore(data)) > threshold).T.tolist()[0]

    # source https://stackoverflow.com/questions/53177358/removing-numbers-which-are-close-to-each-other-in-a-list
    usedValues = set()
    noise_idx_isolated = []
    for v in noise_idx:
        if v not in usedValues:
            noise_idx_isolated.append(v)
            for lv in range(v - padding, v + padding + 1):
                usedValues.add(lv)

    return noise_idx_isolated


def get_features(dset, input, output, idx, padding, size_input, size_future):
    """ Build input and output arrays based on the network architecture."""
    size_past = size_input - size_future
    box_start = idx - padding
    box_end = idx + padding
    for i in range(box_start+size_past, box_end-size_future):
        array = np.array([dset[1, i-size_past:i+size_future],
                          dset[2, i-size_past:i+size_future],
                          dset[3, i-size_past:i+size_future]])
        input.append(array)
    output.append(dset[0, box_start+size_past:box_end-size_future])

    return input, output
