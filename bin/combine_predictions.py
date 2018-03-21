#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_metric_regression')

def combine_predictions(args):
    import numpy as np
    import h5py
    import pandas as pd

    with h5py.File(args.train_test_indices, 'r') as f:
        train_index, test_index = f['train'][:], f['test'][:]
    with h5py.File(args.parent_table, 'r') as f:
        parent_table = f['data'][:]
    region_index = {'s0': np.ravel(parent_table[-5:, -5:]),
                    's1m': np.ravel(parent_table[:-5, -5:]),
                    's1f': np.ravel(parent_table[-5:, :-5])}
    region_index['all'] = np.concatenate(region_index.values())

    predictions = {}
    pred = np.zeros(train_index.shape[0] + test_index.shape[0])

    with h5py.File(os.path.join(args.input_dir, 'trait1.2'), 'r') as f:
        pred[:] = f['y_pred'][:]
    with h5py.File(os.path.join(args.input_dir, 'trait1.3'), 'r') as f:
        pred[region_index['s1m']] = f['y_pred'][region_index['s1m']]
        pred[region_index['s0']] = f['y_pred'][region_index['s0']]
    predictions['trait1'] = pred[test_index]

    with h5py.File(os.path.join(args.input_dir, 'trait2.2'), 'r') as f:
        pred[:] = f['y_pred'][:]
    predictions['trait2'] = pred[test_index]

    with h5py.File(os.path.join(args.input_dir, 'trait3.1'), 'r') as f:
        pred[:] = f['y_pred'][:]
    with h5py.File(os.path.join(args.input_dir, 'trait3.2'), 'r') as f:
        pred[region_index['s1m']] = pred[region_index['s1m']]
    predictions['trait3'] = pred[test_index]

    df = pd.DataFrame(predictions)
    df = df[['trait1', 'trait2', 'trait3']]
    df.to_csv(os.path.join(args.output_dir, 'predictions.txt'), sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine predictions')
    parser.add_argument('--input-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    combine_predictions(args)
