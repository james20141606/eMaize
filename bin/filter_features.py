#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('filter_features')

def open_file_or_stdout(filename):
    if filename == '-':
        return sys.stdout
    else:
        return open(filename, 'r')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def fast_anova_2bit(X, y):
    """
    Fast calculation of ANOVA p-values from 2-bit genotypes
    Each genotype (SNP) is represented by 2 numbers.
        features are grouped by two:
        (SNP1_allele1, SNP1_allele2, SNP2_allele1, SNP2_allele2, ...)
    :param X: 2D array of shape (n_samples, n_snps*2)
    :param y: traits: array of shape (n_samples,)
    :return: ANOVA p-values: 1D array of shape (n_snps,)
    """
    import scipy.stats
    y = y - y.mean()
    y2 = y*y
    N = X.shape[0]
    SS_tot = np.sum(y2)
    # 10, 01, 11
    masks = [np.logical_and(X[:, 0::2], np.logical_not(X[:, 1::2])),
             np.logical_and(np.logical_not(X[:, 0::2]), X[:, 1::2]),
             np.logical_and(X[:, 0::2], X[:, 1::2])]
    Ni = np.concatenate([np.sum(mask, axis=0) for mask in masks]).reshape((3, -1))
    at_least_one = Ni > 0
    SS_bn = [np.sum(y.reshape((-1, 1))*mask, axis=0) for mask in masks]
    SS_bn = np.concatenate(SS_bn).reshape((3, -1))
    SS_bn **= 2
    SS_bn = np.where(at_least_one, SS_bn/Ni, 0)
    SS_bn = np.sum(SS_bn, axis=0)
    SS_wn = SS_tot - SS_bn
    M = np.sum(at_least_one > 0, axis=0)
    DF_bn = M - 1
    DF_wn = N - M
    SS_bn /= DF_bn
    SS_wn /= DF_wn
    F = SS_bn/SS_wn

    p_vals = np.ones(F.shape[0])
    ind = np.nonzero(M == 2)[0]
    if ind.shape[0] > 0:
        p_vals[ind] = scipy.stats.f.sf(F[ind], 1, N - 2)
    ind = np.nonzero(M == 3)[0]
    if ind.shape[0] > 0:
        p_vals[ind] = scipy.stats.f.sf(F[ind], 2, N - 3)
    return p_vals


def fast_linregress(X, y, eps=1e-20):
    """
    Fast computation of linear regression for large number of inputs
    :param X: 2D array of shape (n_features, n_samples)
    :param y: 1D array of shape (n_samples,)
    :param eps: a small number to add to the sum of squares of x
    :return: a tuple (slope, intercept, F-test p-values).
        Each element is of shape (n_samples,).
    """
    N, M = X.shape
    DF = M - 2

    y = y[np.newaxis, :]
    xm = X.mean(axis=1)[:, np.newaxis]
    ym = y.mean()
    X_centered = X - xm
    y_centered = y - ym
    ssx = np.sum(X_centered ** 2, axis=1)[:, np.newaxis]
    ssxy = np.sum(y_centered * X_centered, axis=1)[:, np.newaxis]
    w = ssxy / (ssx + eps)
    b = ym - w * xm
    y_estim = w * X + b

    ssto = np.sum(y_centered ** 2, axis=1)
    sse = np.sum((y - y_estim) ** 2, axis=1)
    ssr = ssto - sse
    f = ssr / (sse / DF)
    pvals_f = scipy.stats.f.sf(f, 1, DF)
    #t = w / (np.sqrt(ssto / DF / ssx))
    #pvals_t =  2.0 * scipy.stats.t.sf(np.abs(t), DF)
    return np.ravel(w), np.ravel(b), np.ravel(pvals_f)

def filter_features_3bit(args):
    logger.info('read sample names from {}'.format(args.sample_names_file))
    with open(args.sample_names_file, 'r') as f:
        sample_names = f.read().strip().split()

    logger.info('read phenotypes from {}'.format(args.phenotype_file))
    phenotypes = pd.read_table(args.phenotype_file)
    n_samples_total = phenotypes.shape[0]
    phenotypes.index = phenotypes['id']
    logger.info('phenotypes: %s' + str(phenotypes.head(3)))

    logger.info('total number of samples: {}'.format(n_samples_total))
    logger.info('read genotypes from {}'.format(args.genotype_file))
    with open(args.genotype_file, 'rb') as f:
        genotypes = np.frombuffer(f.read(), dtype='int8').reshape((-1, n_samples_total))

    genotypes = genotypes[:, phenotypes['id'].isin(sample_names).values]
    phenotypes = phenotypes.ix[sample_names, :]

    p_values = {}
    for trait in ('trait1', 'trait2', 'trait3'):
        p_values[trait] = np.zeros(genotypes.shape[0], dtype='float64')
    for i in xrange(genotypes.shape[0] / 3):
        for trait in ('trait1', 'trait2', 'trait3'):
            samples = []
            for j in range(3):
                ind = np.nonzero(genotypes[i * 3 + j, :])[0]
                if len(ind) > 0:
                    samples.append(phenotypes[trait].values[ind])
            statistic, p_value = scipy.stats.f_oneway(*samples)
            p_values[trait][i] = p_value

    logger.info('save file: %s' % args.outfile)
    prepare_output_file(args.outfile)
    f = h5py.File(args.outfile, 'w')
    for trait in ('trait1', 'trait2', 'trait3'):
        f.create_dataset(trait, data=p_values[trait])
    f.close()

def filter_features_2bit(args):
    from utils import read_hdf5_dataset

    logger.info('read genotypes from: ' + args.genotype_file)
    genotypes = read_hdf5_dataset(args.genotype_file)
    indices = None
    if args.indices_file is not None:
        logger.info('read indices from: ' + args.indices_file)
        indices = read_hdf5_dataset(args.indices_file)
        genotypes = np.take(genotypes, indices, axis=0)
        logger.info('number of samples: %d'%indices.shape[0])
    pvalues = {}
    for phenotype_file in args.phenotype_file:
        logger.info('read phenotypes from: ' + phenotype_file)
        phenotypes, dataset = read_hdf5_dataset(phenotype_file, return_name=True)
        if indices is not None:
            phenotypes = np.take(phenotypes, indices)
        if args.metric == 'anova':
            logger.info('calculate ANOVA p-values')
            pvalues[dataset] = fast_anova_2bit(genotypes, phenotypes)
    logger.info('save p-values to file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        for dataset in pvalues.keys():
            f.create_dataset(dataset, data=pvalues[dataset])

class BatchSliceGenerator(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
        self.n_batches = n//batch_size
        if n % batch_size > 0:
            self.n_batches += 1
    def __call__(self):
        for i in range(self.n_batches):
            yield i*self.batch_size, min((i + 1)*self.batch_size, self.n)

class BatchRunner(object):
    def __init__(self, func, n, batch_size, slice_dims=None, concat_output=True):
        self.func = func
        self.n = n
        self.batch_size = batch_size
        self.slice_dims = slice_dims
        self.n_batches = int(round(float(n)/batch_size))
    def __call__(self, *args, **kwargs):
        outputs = []
        for i in range(self.n_batches):
            indices = []
            for slice_dim in self.slice_dims:
                indices.append(slice(i*self.batch_size, min((i + 1)*self.batch_size, self.n), None))
            outputs.append(self.func())

def anova_linregress(args):
    from utils import read_hdf5_dataset
    from tqdm import tqdm
    from statsmodels.sandbox.stats.multicomp import multipletests

    logger.info('read genotypes from: ' + args.genotype_file)
    genotypes = read_hdf5_dataset(args.genotype_file)
    indices = None
    if args.sample_indices_file is not None:
        logger.info('read indices from: ' + args.sample_indices_file)
        indices = read_hdf5_dataset(args.sample_indices_file)
        genotypes = np.take(genotypes, indices, axis=1)
        logger.info('number of samples: %d' % indices.shape[0])

    logger.info('read phenotypes from: ' + args.phenotype_file)
    phenotypes, dataset = read_hdf5_dataset(args.phenotype_file, return_name=True)
    logger.info('perform ANOVA for dataset: %s'%dataset)
    if indices is not None:
        phenotypes = np.take(phenotypes, indices)
    if args.batch_size is not None:
        slicegen = BatchSliceGenerator(genotypes.shape[0], batch_size=args.batch_size)
        outputs = []
        for start, stop in tqdm(slicegen(), total=slicegen.n_batches):
            outputs.append(fast_linregress(genotypes[start:stop], phenotypes))
        w, b, pvalues = [np.concatenate(a) for a in zip(*outputs)]
        del outputs
    else:
        w, b, pvalues = fast_linregress(genotypes, phenotypes)
    reject, qvalues, _, _ = multipletests(pvalues, alpha=args.alpha, method='fdr_bh')
    reject = np.nonzero(reject)[0]

    logger.info('save results to file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('pvalue', data=pvalues)
        f.create_dataset('slope', data=w.astype('float32'))
        f.create_dataset('intercept', data=b.astype('float32'))
        f.create_dataset('qvalue', data=qvalues)
        f.create_dataset('reject', data=reject)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Predict using saved models')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('3bit_binary')
    parser.add_argument('--genotype-file', '-i', type=str, required=True)
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('--sample-names-file', type=str, required=True)
    parser.add_argument('--outfile', '-o', type=str, required=True)
    parser.add_argument('--metric', type=str, default='anova')

    parser = subparsers.add_parser('2bit')
    parser.add_argument('--genotype-file', '-i', type=str, required=True,
                        help='HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--phenotype-file', type=str, required=True, nargs='+',
                        help='HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--indices-file', type=str,
                        help='indices to select. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--output-file', '-o', type=str, required=True)
    parser.add_argument('--metric', type=str, default='anova')

    parser = subparsers.add_parser('anova_linregress')
    parser.add_argument('--genotype-file', '-i', type=str, required=True,
                        help='HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--sample-indices-file', type=str,
                        help='indices to select. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--batch-size', type=int,
                        help='number of features to process in a batch')
    parser.add_argument('--alpha', type=float, default=1e-4,
                        help='FWER for multiple test correction (Benjamini/Hochberg method)')
    parser.add_argument('--output-file', '-o', type=str, required=True)


    args = main_parser.parse_args()

    import numpy as np
    import pandas as pd
    import h5py
    import scipy.stats

    if args.command == '3bit_binary':
        filter_features_3bit(args)
    elif args.command == '2bit':
        filter_features_2bit(args)
    elif args.command == 'anova_linregress':
        anova_linregress(args)

