# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src import utils
from pathlib import Path
from sklearn.model_selection import train_test_split as tt_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay as ITD


parser = argparse.ArgumentParser()
parser.add_argument("--indir",
                    help="input directory, default: data/processed",
                    default=Path('data/processed'))
parser.add_argument("--outdir",
                    help="output directory, default: models/GW200129",
                    default=Path('models/GW200129'))
parser.add_argument("--model", help="ML model, default: DNN", default='DNN',
                    type=str)
parser.add_argument("--train_mode", help=("ML training mode, 'fast' for "
                                          "exploratory runs or 'slow' for "
                                          "deployment. Default: 'slow'"),
                    default='slow', type=str)
args = parser.parse_args()


def build_train_dnn(train_input, train_output, val_input, val_output, outdir,
                    mode):
    # model params
    width = 1024
    width = 10
    l2 = 1e-4
    dropout = 0.2

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(width, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(width, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dense(1),
    ])

    if mode == 'fast':
        lr = 1e-4
        epochs = 50
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=lr))
    elif mode == 'slow':
        lr = 1e-3
        epochs = 500
        lr_schedule = ITD(lr, decay_steps=80, decay_rate=10, staircase=True)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=lr_schedule))

    batch = 10000
    model_fit = model.fit(train_input, train_output,
                          validation_data=(val_input, val_output),
                          batch_size=batch, epochs=epochs, verbose=1)

    model.save(outdir)

    plt.figure()
    plt.semilogy(model_fit.history['loss'], label='training')
    plt.semilogy(model_fit.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.savefig(f'{outdir}/training_hist.png', bbox_inches='tight')
    plt.close()


def main(indir, outdir, type_, mode):
    """ Train a model. """
    logger = logging.getLogger(__name__)
    logger.info('Starting model training')

    indir = Path(indir)
    utils.chdir(indir, logger)
    outdir = Path(outdir)
    utils.chdir(outdir, logger, create=True)

    logger.info('Loading features')
    features = np.load(f'{indir}/features.npz')

    seed = 0
    split = 0.1
    train_input, val_input = tt_split(features['input'],
                                      test_size=split,
                                      shuffle=True,
                                      random_state=seed)
    train_output, val_output = tt_split(features['output'],
                                        test_size=split,
                                        shuffle=True,
                                        random_state=seed)

    logger.info('Training the model')
    if type_ == 'DNN':
        build_train_dnn(train_input, train_output,
                        val_input, val_output,
                        outdir, mode)
    else:
        logger.error("Unknown model architecture")
        raise SystemExit(1)

    np.savez(f'{outdir}/dataset_params',
             norm=features['norm'],
             fs=features['fs'],
             size_input=features['size_input'],
             size_future=features['size_future'])
    features.close()
    logger.info('Model training finished')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(args.indir, args.outdir, args.model, args.train_mode)
