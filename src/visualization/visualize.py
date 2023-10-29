#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import pdb
import matplotlib.pyplot as plt
from src import utils
from pathlib import Path
from gwpy.timeseries import TimeSeries


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration, default: GW200129",
                    default='GW200129', type=str)
parser.add_argument("--noisydir",
                    help="directory of the noisy timeseries, default: data/external",
                    default=Path('data/external'))
parser.add_argument("--cleandir", help="cirectory of the clean timeseris, default: data/predicted",
                    default=Path('data/predicted'), type=str)
parser.add_argument("--outdir",
                    help="output directory for visualizations, default: reports/figures",
                    default=Path('reports/figures'))
args = parser.parse_args()


def make_oscan(noisy, clean, t0, win_plot, win_crop, alim, ylim, qrange, outdir):
    noisy = noisy.crop(t0 - win_crop/2, t0 + win_crop/2)
    clean = clean.crop(t0 - win_crop/2, t0 + win_crop/2)

    noisy = TimeSeries(noisy, times=noisy.times.value - t0)
    clean = TimeSeries(clean, times=clean.times.value - t0)

    plot_start = t0 - win_plot/2
    plot_end = t0 + win_plot/2

    dataset = ['orig', 'clean', 'diff']
    q_trans = {}
    q_trans['orig'] = noisy.q_transform(outseg=(plot_start, plot_end), qrange=qrange)
    q_trans['clean'] = clean.q_transform(outseg=(plot_start, plot_end), qrange=qrange)
    q_trans['diff'] = q_trans['orig'] - q_trans['clean']

    label = {}
    label['orig'] = 'Original data'
    label['clean'] = 'Cleaned data'
    label['diff'] = 'Original - Cleaned'

    pdb.set_trace()
    plot, axes = plt.subplots(nrows=3, sharex=True, figsize=(3.375*2.0, 3.375*3.0))

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

    plot.tight_layout()
    plot.subplots_adjust(top=0.85)
    plot.savefig(f'{outdir}/{t0}.png', dpi=400)


def main(config, ndir, cdir, outdir):
    """ Producing noisy vs cleaned spectrograms. """
    logger = logging.getLogger(__name__)
    logger.info('Making spectrograms')

    ndir = Path(ndir)
    utils.chdir(ndir, logger)
    cdir = Path(cdir)
    utils.chdir(cdir, logger)
    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    if config == 'GW200129':
        noisy = TimeSeries.read(f'{ndir}/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_event.hdf5')
        clean = TimeSeries.read(f'{cdir}/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_NLSUB.hdf5')

        win_plot = 4
        win_crop = 4
        alim = (0, 25)
        ylim = (10, 1024)
        qrange = (10, 20)
        times = [1264316116, 1264316154, 1264316164]
        for time in times:
            make_oscan(noisy, clean, time, win_plot, win_crop, alim, ylim, qrange, outdir)
        logger.info('Spectrograms produced.')

    else:
        logger.error('Unknown config')
        raise SystemExit(1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.config, args.noisydir, args.cleandir, args.outdir)
