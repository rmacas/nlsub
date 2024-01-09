#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
from src import utils
from pathlib import Path
from gwpy.timeseries import TimeSeries


parser = argparse.ArgumentParser()
parser.add_argument("--noisy",
                    help="noisy timeseries, default: data/external/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_event.hdf5",  # noqa: E501
                    default=Path('data/external/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_event.hdf5'))  # noqa: E501
parser.add_argument("--clean",
                    help="cleaned timeseries, default: data/predicted/DCS-CALIB_STRAIN_NLSUB_C01_4096Hz_event.hdf5",  # noqa: E501
                    default=Path('data/predicted/DCS-CALIB_STRAIN_NLSUB_C01_4096Hz_event.hdf5'))  # noqa: E501
parser.add_argument("--outdir",
                    help="outdir for visualizations, default: reports/figures",
                    default=Path('reports/figures'))
args = parser.parse_args()


def main(noisy, clean, outdir):
    """ Producing noisy vs cleaned spectrograms. """
    logger = logging.getLogger(__name__)
    logger.info('Making spectrograms')

    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    n = TimeSeries.read(noisy)
    c = TimeSeries.read(clean)

    times = [1264316116.345, 1264316154, 1264316164]
    for time in times:
        utils.make_oscan(n, c, time, outdir)
    logger.info('Spectrograms produced')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.noisy, args.clean, args.outdir)
