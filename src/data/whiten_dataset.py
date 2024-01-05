# -*- coding: utf-8 -*-
import argparse
import logging
from src import utils
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    help="input directory, default: data/external",
                    default=Path('data/external'))
parser.add_argument("--outdir",
                    help="output directory, default: data/interim",
                    default=Path('data/interim'))
args = parser.parse_args()


def whiten(outdir, tname, fname, channel):
    """ Whiten the TimeSeries given FrequencySeries ASD."""
    tseries = TimeSeries.read(tname)
    asd = FrequencySeries.read(fname)
    twhite = tseries.whiten(asd=asd)

    tname = tname.split('/')[-1].split('.')[0]
    twhite.write(f'{outdir}/{tname}_whitened.hdf5', path=channel)
    return


def main(indir, outdir):
    """ Data whitening."""
    logger = logging.getLogger(__name__)
    logger.info('Whitening data')

    indir = Path(indir)
    utils.chdir(indir, logger)
    outdir = Path(outdir)
    utils.chdir(outdir, logger)

    fs = 512
    suffixes = ['27hrs', 'event']
    channels = ['L1:DCS-CALIB_STRAIN_CLEAN_C01',
                'L1:LSC-POP_A_RF45_I_ERR_DQ',
                'L1:LSC-POP_A_RF45_Q_ERR_DQ',
                'L1:LSC-POP_A_RF9_I_ERR_DQ']

    for suffix in suffixes:
        for channel in channels:
            tname = f'{indir}/{channel[3:]}_{fs}Hz_{suffix}.hdf5'
            fname = f'{indir}/{channel[3:]}_{fs}Hz_ASD.hdf5'
            whiten(outdir, tname, fname, channel)

    logger.info('Data whitened')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args.indir, args.outdir)
