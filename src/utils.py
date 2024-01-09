#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from scipy.stats import zscore


def chdir(dir_, logger, create=False):
    """Directory check; optionally create the directory."""
    if not dir_.is_dir():
        if create:
            logger.warning(f"The directory {dir_} doesn't exist. Creating it")
            dir_.mkdir(parents=True, exist_ok=True)
        else:
            logger.error(f"The directory {dir_} doesn't exist")
            raise SystemExit(1)


def get_tseries(outdir, suffix, frame, channel, fs, gps_start, gps_end):
    """Get time series using GWpy. Needs access to LIGO proprietary data."""
    tseries = TimeSeries.get(channel=channel, start=gps_start, end=gps_end,
                             frametype=frame).resample(fs)
    tseries.write(f'{outdir}/{channel[3:]}_{fs}Hz_{suffix}.hdf5',
                  overwrite=True, path=channel)
    return


def get_asd(outdir, frame, channel, fs, gps_start, gps_end):
    """Get ASD using GWpy. Needs access to LIGO proprietary data."""
    tseries = TimeSeries.get(channel=channel, start=gps_start, end=gps_end,
                             frametype=frame).resample(fs)
    asd = tseries.asd(4, 2, method='median')
    asd.write(f'{outdir}/{channel[3:]}_{fs}Hz_ASD.hdf5', overwrite=True,
              path=channel)
    return


def whiten(outdir, tname, fname, channel):
    """ Whiten the TimeSeries given FrequencySeries ASD."""
    tseries = TimeSeries.read(tname)
    asd = FrequencySeries.read(fname)

    fseries = scipy.fft.rfft(tseries)

    freqs = asd.frequencies
    freqs_interp = np.linspace(freqs[0], freqs[-1], len(fseries))
    asd_interp = np.interp(freqs_interp, freqs, asd.value)

    whitened = scipy.fft.irfft(fseries/asd_interp)

    twhite = TimeSeries(whitened, times=tseries.times)

    tname = tname.split('/')[-1].split('.')[0]
    twhite.write(f'{outdir}/{tname}_whitened.hdf5', overwrite=True,
                 path=channel)
    return


def load_data(indir, channels, fs):
    """Loading time-series data."""
    dset = []
    norm = []
    for channel in channels:

        fname = f'{indir}/{channel}'
        tseries = TimeSeries.read(fname)

        # remove sides due to whitening artifacts
        data = tseries[fs*4:-fs*4].reshape(-1, 1)
        time = tseries[fs*4:-fs*4].times

        # normalize and record the normalisation factor
        norm_factor = np.max(np.abs(data))
        norm.append(norm_factor)
        dset.append(data / norm_factor)

    dset = np.squeeze(dset).astype('float32')
    time = np.array(time)

    return dset, time, norm


def find_noisy(data, threshold, padding):
    """ Find idx passing the Z-score threshold."""

    noise_idx = np.argwhere(np.abs(zscore(data)) > threshold).T.tolist()[0]

    # source https://stackoverflow.com/questions/53177358/removing-numbers-which-are-close-to-each-other-in-a-list  # noqa: E501
    usedValues = set()
    noise_idx_isolated = []
    for v in noise_idx:
        if v not in usedValues:
            noise_idx_isolated.append(v)
            for lv in range(v - padding, v + padding + 1):
                usedValues.add(lv)

    return noise_idx_isolated


def get_features(dset, input, output, box_start, box_end, size_input,
                 size_future):
    """ Build input and output arrays based on the network architecture."""
    size_past = size_input - size_future
    for i in range(box_start+size_past, box_end-size_future):
        array = np.array([dset[1, i-size_past:i+size_future],
                          dset[2, i-size_past:i+size_future],
                          dset[3, i-size_past:i+size_future]])
        input.append(array)
    output.append(dset[0, box_start+size_past:box_end-size_future])

    return input, output


def make_oscan(noisy, clean, t0, outdir):
    """Plot omegascans of the original, cleaned and the difference of the
    time-series.
    """
    noisy = TimeSeries(noisy, times=noisy.times.value - t0)
    clean = TimeSeries(clean, times=clean.times.value - t0)

    win_crop = 40
    win_plot = 4
    plot_start = -win_plot/2
    plot_end = win_plot/2
    ylim = (10, 512)
    alim = (0, 25)
    qrange = (10, 20)

    noisy = noisy.crop(-win_crop/2, win_crop/2)
    clean = clean.crop(-win_crop/2, win_crop/2)

    dataset = ['orig', 'clean', 'diff']
    q_trans = {}
    q_trans['orig'] = noisy.q_transform(outseg=(plot_start, plot_end),
                                        qrange=qrange)
    q_trans['clean'] = clean.q_transform(outseg=(plot_start, plot_end),
                                         qrange=qrange)
    q_trans['diff'] = q_trans['orig'] - q_trans['clean']

    label = {}
    label['orig'] = 'Original data'
    label['clean'] = 'Cleaned data'
    label['diff'] = 'Original - Cleaned'

    plot, axes = plt.subplots(nrows=3, sharex=True, figsize=(3.375*2, 3.375*3))

    for i, ax in zip(dataset, axes):
        ax.imshow(q_trans[i], vmin=alim[0], vmax=alim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel('')
        ax.set_yscale('log')
        ax.plot([0], 10, label=label[i], visible=False)
        ax.grid(alpha=0.6)
        ax.legend(loc='upper left', handlelength=0, handletextpad=0)

    axes[1].set_ylabel(r"$\mathrm{Frequency \ (Hz)}$")
    axes[-1].set_xlabel(r"$\mathrm{Time \ (seconds)}$")
    cbar = axes[0].colorbar(clim=(alim[0], alim[1]), location='top')
    cbar.set_label(r"$\mathrm{Normalized \ energy}$")
    plot.savefig(f'{outdir}/{t0}.png', dpi=400)
