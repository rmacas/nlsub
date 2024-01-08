#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
from src import utils
from pathlib import Path
from gwpy.timeseries import TimeSeries


parser = argparse.ArgumentParser()
parser.add_argument("--noisydir",
                    help="noisy timeseries directory, default: data/external",
                    default=Path('data/external'))
parser.add_argument("--cleandir",
                    help="clean timeseries directory, default: data/predicted",
                    default=Path('data/predicted'))
parser.add_argument("--outdir",
                    help="outdir for visualizations, default: reports/figures",
                    default=Path('reports/figures'))
args = parser.parse_args()


def main(ndir, cdir, outdir):
    """ Producing noisy vs cleaned spectrograms. """
    logger = logging.getLogger(__name__)
    logger.info('Making spectrograms')

    ndir = Path(ndir)
    utils.chdir(ndir, logger)
    cdir = Path(cdir)
    utils.chdir(cdir, logger)
    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    n = TimeSeries.read(f'{ndir}/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_event.hdf5')
    c = TimeSeries.read(f'{cdir}/DCS-CALIB_STRAIN_NLSUB_C01_4096Hz_event.hdf5')

    times = [1264316116, 1264316154, 1264316164]
    for time in times:
        utils.make_oscan(n, c, time, outdir)
    logger.info('Spectrograms produced')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.noisydir, args.cleandir, args.outdir)
