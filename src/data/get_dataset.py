# -*- coding: utf-8 -*-
import argparse
import logging
from src import utils
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--outdir",
                    help="output directory, default: data/external",
                    default=Path('data/external'))
args = parser.parse_args()


def main(outdir):
    """ Runs a script to get minimally processed timeseries and ASD data from
    LIGO. Requires internal access to LIGO proprietary data. Takes *hours*.
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading data')

    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    fs = 512  # sampling rate

    frame = 'L1_HOFT_C01'
    channels = ['L1:DCS-CALIB_STRAIN_CLEAN_C01',
                'L1:LSC-POP_A_RF45_I_ERR_DQ',
                'L1:LSC-POP_A_RF45_Q_ERR_DQ',
                'L1:LSC-POP_A_RF9_I_ERR_DQ']

    # get data around the event
    suffix = 'event'
    gps_event = 1264316116.435
    gps_start = gps_event - 2048
    gps_end = gps_event + 2048

    for channel in channels:
        utils.get_tseries(outdir, suffix, frame, channel, fs, gps_start,
                          gps_end)

    # get 4096 Hz data used to get the cleaned frame
    utils.get_tseries(outdir, suffix, frame, channels[0], 4096, gps_start,
                      gps_end)

    # get 27hr data used for training
    suffix = '27hrs'
    gps_start = 1264326924
    gps_end = gps_start + 27*60*60

    for channel in channels:
        utils.get_tseries(outdir, suffix, frame, channel, fs, gps_start,
                          gps_end)

    # get ASD for data whitening
    gps_asd = gps_event - 450
    asd_win = 256
    gps_start = gps_asd - asd_win
    gps_end = gps_asd + asd_win

    for channel in channels:
        utils.get_asd(outdir, frame, channel, fs, gps_start, gps_end)

    logger.info('Data downloaded')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.outdir)
