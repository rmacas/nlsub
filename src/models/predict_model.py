#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import scipy
import numpy as np
import tensorflow as tf
from src import utils
from pathlib import Path
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration, default: GW200129",
                    default='GW200129', type=str)
parser.add_argument("--whdir",
                    help="whitened data directory, default: data/interim",
                    default=Path('data/interim'))
parser.add_argument("--asddir",
                    help="ASD directory, default: data/external",
                    default=Path('data/external'))
parser.add_argument("--modeldir", help="ML model directory, default: GW200129",
                    default=Path('models/GW200129'), type=str)
parser.add_argument("--outdir",
                    help="output directory, default: data/predicted",
                    default=Path('data/predicted'))
parser.add_argument("--mem-reduction", default=32, type=int,
                    help=('Reduces the memory requirement by --mem-reduction '
                          'factor but slows down the prediction.'))
args = parser.parse_args()


def load_data(indir, channels, fs):

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


def get_iarray(dset, input, box_start, box_end, size_input, size_future):
    """ Build input array based on the network architecture."""
    size_past = size_input - size_future
    for i in range(box_start+size_past, box_end-size_future):
        array = np.array([dset[1, i-size_past:i+size_future],
                          dset[2, i-size_past:i+size_future],
                          dset[3, i-size_past:i+size_future]])
        input.append(array)
    return input


def main(config, whdir, asddir, modeldir, outdir, mem):
    """ Use the model to get cleaned frame. """
    logger = logging.getLogger(__name__)
    logger.info('Starting the frame cleaning')
    whdir = Path(whdir)
    utils.chdir(whdir, logger)
    asddir = Path(asddir)
    utils.chdir(asddir, logger)
    modeldir = Path(modeldir)
    utils.chdir(modeldir, logger)
    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    if config == 'GW200129':
        logger.info('Loading and preparing the data for GW200129')

        with np.load(f'{modeldir}/dataset_params.npz') as params:
            norm_orig = params['norm']
            fs = params['fs']
            size_input = params['size_input']
            size_future = params['size_future']

        channels = ['L1:DCS-CALIB_STRAIN_CLEAN_C01',
                    'L1:LSC-POP_A_RF45_I_ERR_DQ',
                    'L1:LSC-POP_A_RF45_Q_ERR_DQ',
                    'L1:LSC-POP_A_RF9_I_ERR_DQ']

        asd = FrequencySeries.read(f'{asddir}/{channels[0][3:]}_{fs}Hz_ASD.hdf5')

        suffix = f'{fs}Hz_event_whitened.hdf5'
        channels = [f'{channel[3:]}_{suffix}' for channel in channels]
        dset, time, norm = load_data(whdir, channels, fs)

        # re-normalize the data w.r.t. the dataset used in the training
        for i in range(len(dset)):
            dset[i, :] = dset[i, :] * norm[i] / norm_orig[i]

        logger.info('Using the model to predict the noise')
        model = tf.keras.models.load_model(modeldir)

        # splitting the data in smaller chunks to reduce memory requirements
        breaks = np.linspace(0, dset.shape[1], num=mem, endpoint=True, dtype=int)
        prediction = []
        for i in range(len(breaks)-1):
            input = []
            box_start = 0 if i == 0 else breaks[i] - size_input
            box_end = breaks[i+1]
            input = get_iarray(dset, input, box_start, box_end, size_input, size_future)
            prediction.append(model.predict(np.array(input)))

        prediction = np.squeeze(np.vstack(prediction))
        time = time[size_input-size_future:-size_future]

        logger.info('Upsampling and coloring the noise')
        f_low = 10
        asd_win = 4
        firwin = scipy.signal.firwin(asd_win*fs+1, [f_low], pass_zero=False, window='hann', fs=fs)
        ffirwin = np.abs(scipy.fft.rfft(firwin))
        asd = ffirwin * asd

        time_asd = scipy.fft.irfft(asd)
        time_asd = np.roll(time_asd, len(time_asd)//2)
        hann = scipy.signal.windows.hann(len(time_asd))
        time_asd = time_asd * hann

        time_asd = np.pad(time_asd, (0, len(prediction)-len(time_asd)))
        freq_asd = np.abs(scipy.fft.rfft(time_asd))

        fseries = scipy.fft.rfft(prediction)

        colored = scipy.fft.irfft(fseries * freq_asd) * norm_orig[0]

        colored = TimeSeries(colored[fs*4:-fs*4], times=time[fs*4:-fs*4])

        fs_new = 4096
        colored = colored.resample(fs_new)

        channel_noisy = 'DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_event.hdf5'
        tseries_noisy = TimeSeries.read(f'{asddir}/{channel_noisy}')
        tseries_noisy = tseries_noisy.crop(colored.times[0], colored.times[-1])

        tseries_clean = tseries_noisy - colored[:-1]
        channel_clean = 'DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_NLSUB.hdf5'
        tseries_clean.write(f'{outdir}/{channel_clean}', path=channel_clean)
        logger.info('Frame cleaned')

    else:
        logger.error('Unknown config')
        raise SystemExit(1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.config, args.whdir, args.asddir, args.modeldir, args.outdir,
         args.mem_reduction)
