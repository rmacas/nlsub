# -*- coding: utf-8 -*-
import argparse
import logging
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration, default: GW200129",
                    default='GW200129', type=str)
parser.add_argument("--indir",
                    help="input directory, default: data/external",
                    default=Path('data/external'))
parser.add_argument("--outdir",
                    help="output directory, default: data/interim",
                    default=Path('data/interim'))
args = parser.parse_args()


def whiten(outdir, tname, fname, channel):
    """ White the TimeSeries given FrequencySeries ASD."""
    tseries = TimeSeries.read(tname)
    asd = FrequencySeries.read(fname)
    twhite = tseries.whiten(asd=asd)

    tname = tname.split('/')[-1].split('.')[0]
    twhite.write(f'{outdir}/{tname}_whitened.hdf5', path=channel)
    return


def main(config, indir, outdir):
    """ Data whitening."""
    logger = logging.getLogger(__name__)
    logger.info('Making final data set to be used for feature extraction')

    indir = Path(indir)
    if not indir.is_dir():
        logger.error("The input directory doesn't exist")
        raise SystemExit(1)
    outdir = Path(outdir)
    if not outdir.is_dir():
        logger.error("The output directory doesn't exist")
        raise SystemExit(1)

    if config == 'GW200129':
        logger.info('Whitening data for GW200129')

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

        # do this separately for the 4096Hz original frame
        tname = f'{indir}/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_event.hdf5'
        fname = f'{indir}/DCS-CALIB_STRAIN_CLEAN_C01_4096Hz_ASD.hdf5'
        whiten(outdir, tname, fname, channel)

        logger.info('Data whitened')

    else:
        logger.error('Unknown config')
        raise SystemExit(1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(args.config, args.indir, args.outdir)
