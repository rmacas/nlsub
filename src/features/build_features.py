# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
from pathlib import Path
from src import utils


parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    help="input directory, default: data/interim",
                    default=Path('data/interim'))
parser.add_argument("--outdir",
                    help="output directory, default: data/processed",
                    default=Path('data/processed'))
args = parser.parse_args()


def main(indir, outdir):
    """ Feature extraction to train a ML model on."""
    logger = logging.getLogger(__name__)
    logger.info('Extracting features from the data set')

    indir = Path(indir)
    utils.chdir(indir, logger)
    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    logger.info('Loading data')

    fs = 512  # sampling rate
    suffix = '512Hz_27hrs_whitened.hdf5'
    channels = ['L1:DCS-CALIB_STRAIN_CLEAN_C01',
                'L1:LSC-POP_A_RF45_I_ERR_DQ',
                'L1:LSC-POP_A_RF45_Q_ERR_DQ',
                'L1:LSC-POP_A_RF9_I_ERR_DQ']
    channels = [f'{channel[3:]}_{suffix}' for channel in channels]

    dset, _, norm = utils.load_data(indir, channels, fs)

    logger.info('Processing data')
    logger.warning('This is a memory-intensive task. Requires O(20GB) '
                   'memory')
    padding = fs * 4  # amount of data considered 'related' around an idx
    threshold = 11.7  # found by trial for dset[1]
    noisy_idx = utils.find_noisy(dset[1], threshold, padding)

    # input array params
    size_input = int(fs*1.5)
    size_future = int(fs/2)
    size_input = 768
    size_future = 256

    # get features
    input = []
    output = []
    for idx in noisy_idx:
        box_start = idx - padding
        box_end = idx + padding
        input, output = utils.get_features(dset, input, output, box_start,
                                           box_end, size_input, size_future)
    input = np.array(input)
    output = np.hstack(output).reshape(-1, 1)

    # remove glitches
    glitch_threshold = 40  # threshold factor found by trial
    glitch_idx = utils.find_noisy(output, glitch_threshold, padding)

    for idx in glitch_idx[::-1]:
        idx_start = idx - padding - size_input - size_future
        idx_end = idx + padding + size_future
        output = np.delete(output, slice(idx_start, idx_end), axis=0)
        input = np.delete(input, slice(idx_start, idx_end), axis=0)

    # normalisation
    norm_output = np.max(np.abs(output))
    output = output / norm_output
    norm[0] *= norm_output
    # input array has max abs val of 1, no need to normalise it

    np.savez_compressed(f'{outdir}/features',
                        input=input, output=output, norm=norm, fs=fs,
                        size_input=size_input, size_future=size_future)
    logger.info('Features extracted')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.indir, args.outdir)
