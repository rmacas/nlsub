# -*- coding: utf-8 -*-
import argparse
import logging
from src import utils
from gwpy.timeseries import TimeSeries
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration, default: GW200129",
                    default='GW200129', type=str)
parser.add_argument("--outdir",
                    help="output directory, default: data/external",
                    default=Path('data/external'))
args = parser.parse_args()


def get_tseries(outdir, suffix, frame, channel, fs, gps_start, gps_end):
    """Get time series using GWpy. Needs access to LIGO proprietary data."""
    tseries = TimeSeries.get(channel=channel, start=gps_start, end=gps_end,
                             frametype=frame).resample(fs)
    tseries.write(f'{outdir}/{channel[3:]}_{fs}Hz_{suffix}.hdf5', path=channel)
    return


def get_asd(outdir, frame, channel, fs, gps_start, gps_end):
    """Get ASD using GWpy. Needs access to LIGO proprietary data."""
    tseries = TimeSeries.get(channel=channel, start=gps_start, end=gps_end,
                             frametype=frame).resample(fs)
    asd = tseries.asd(4, 2, method='median')
    asd.write(f'{outdir}/{channel[3:]}_{fs}Hz_ASD.hdf5', path=channel)
    return


def main(config, outdir):
    """ Runs a script to get minimally processed timeseries and ASD data from
    LIGO. Requires internal access to LIGO proprietary data. Takes *a lot* of
    time if the data is on tape.
    """
    logger = logging.getLogger(__name__)

    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    if config == 'GW200129':
        logger.info('Downloading data for GW200129')

        # sampling rate
        fs = 512

        # frame and channels
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
            get_tseries(outdir, suffix, frame, channel, fs, gps_start, gps_end)

        # get 4096 Hz frame used to get cleaned frame
        get_tseries(outdir, suffix, frame, channels[0], 4096, gps_start,
                    gps_end)

        # get 27hr data used for training
        suffix = '27hrs'
        gps_start = 1264326924
        gps_end = gps_start + 27*60*60

        for channel in channels:
            get_tseries(outdir, suffix, frame, channel, fs, gps_start, gps_end)

        # get ASD
        gps_asd = gps_event - 450
        asd_win = 256
        gps_start = gps_asd - asd_win
        gps_end = gps_asd + asd_win

        for channel in channels:
            get_asd(outdir, frame, channel, fs, gps_start, gps_end)

        logger.info('Data downloaded')

    else:
        logger.error('Unknown config')
        raise SystemExit(1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.config, args.outdir)
