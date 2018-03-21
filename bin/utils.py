import os, errno
import numpy as np
import scipy.stats
import numba
from scipy.stats import linregress
from tqdm import tqdm

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def read_hdf5_single(filename, return_name=False):
    """
    Read a dataset from an HDF5 file that contains a single dataset
    :param filename: path of the HDF5 file
    :param return_name: whether to return the dataset name
    :return: the dataset
    """
    import h5py
    with h5py.File(filename, 'r') as f:
        if len(f.keys()) == 0:
            raise ValueError('the HDF5 file is empty: ' + filename)
        dataset = f.keys()[0]
        data = f[dataset][:]
    if return_name:
        return data, dataset
    else:
        return data

def read_hdf5_dataset(filename, return_name=False):
    """
    Read a dataset from an HDF5 file
    :param filename: file path and dataset name separated by ":" (e.g file.h5:dataset)
    :return: the dataset
    """
    import h5py
    if ':' not in filename:
        raise ValueError('missing dataset name in the HDF5 file: ' + filename)
    i = filename.index(':')
    f = h5py.File(filename[:i], 'r')
    dataset = filename[(i + 1):]
    data = f[dataset][:]
    f.close()
    if return_name:
        return data, dataset
    else:
        return data

def read_cvindex(filename):
    import h5py
    if ':' not in filename:
        raise ValueError('missing dataset name in the HDF5 file: ' + filename)
    i = filename.index(':')
    group = filename[(i + 1):]
    filename = filename[:i]
    with h5py.File(filename, 'r') as f:
        g = f[group]
        train_index = g['train'][:]
        test_index = g['test'][:]
    return train_index, test_index

def generate_parent_table(phenotype_file):
    """
    Organize the sample indices into a table by the pedigree relationship.
    :param phenotype_file: pheno_emaize.txt
    :return: a matrix of sample indices of shape (n_males, n_females)
    """
    import pandas as pd
    import numpy as np

    phenotypes = pd.read_table(phenotype_file)
    pedigree = phenotypes['pedigree'].str.split('_', expand=True)
    pedigree.columns = ['f', 'X', 'm']
    phenotypes = pd.concat([phenotypes, pedigree], axis=1)
    phenotypes['number'] = np.arange(phenotypes.shape[0])
    parent_table = phenotypes.pivot_table(values='number', index=['m'], columns=['f'], dropna=False)
    male_ids = ['m%d' % i for i in range(1, parent_table.shape[0] + 1)]
    female_ids = ['f%d' % i for i in range(1, parent_table.shape[1] + 1)]
    parent_table = parent_table.loc[male_ids, female_ids].values
    return parent_table

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

@numba.jit
def take2d(x, indices1, indices2):
    res = np.zeros((indices1.shape[0], indices2.shape[0]))
    for i in range(indices1.shape[0]):
        for j in range(indices2.shape[0]):
            res[i, j] = x[indices1[i], indices2[j]]
    return res

def standardize_genotypes(X):
    '''
    Normalize features to zero means and unit variances.
    :param X: 2D array of shape [n_samples, n_snps]
    :return: 2D array of shape [n_samples, n_snps]
    '''
    X = X.astype('float64')
    X -= X.mean(axis=0)[np.newaxis, :]
    X_std = np.sqrt(np.sum(X*X, axis=0))
    X_std[X_std == 0] = 1.0
    X /= X_std[np.newaxis, :]
    return X

def get_indices_table(sample_indices, parent_table):
    """
    Map sample indices to a table specified by parent table.
    Remove rows and columns with no samples.
    Map sample indices to indices that starts with 0 and ends with len(sample_indices) -1.

    :param sample_indices: 1D array of sample indices
    :param parent_table:
    :return: (indices_table, mask)
        indices_table: 2D array of sample indices (range from 0 to len(sample_indices) - 1)
        mask: an 1D boolean array
    """
    n_samples_total = np.prod(parent_table.shape)
    notnan_table = np.zeros(n_samples_total, dtype='bool')
    notnan_table[sample_indices] = True
    notnan_table = np.take(notnan_table, parent_table)
    indices_table = np.full(n_samples_total, len(sample_indices), dtype='int64')
    indices_table[sample_indices] = np.arange(len(sample_indices))
    indices_table = np.take(indices_table, parent_table)

    notnan_row = np.nonzero(np.any(notnan_table, axis=1))[0]
    notnan_col = np.nonzero(np.any(notnan_table, axis=0))[0]
    indices_table = indices_table[notnan_row, :][:, notnan_col]
    # NaNs are indicated by the length of sample_indices
    mask = np.ones(len(sample_indices) + 1, dtype='bool')
    mask[-1] = False
    return indices_table, mask

def cv_split_emaize(indices_table, mask=None, k=None, k1=3, k2=3, shuffle=False, method='cross'):
    """
    Generate cross-validation indices.
    For each CV fold, the training samples are a contiguous block in indices_table, and
    the test samples are the remaining samples (consist of three blocks).
    :param indices_table: a 2D array of indices. Shape: [n_rows, n_cols]
    :param mask: a 1D boolean array that indicates whether to include an index in indices_table.
        Mask indices correpond to sample indices.
    :param k: number of folds, will override k1 and k2
    :param k1: number of folds for row split
    :param k2: number of folds for column split
    :param shuffle: shuffle the rows and columns before splitting
    :param method: s0: unknown females and males. s1f: known females (by row). s1m: known males (by col).
    :return: a tuple (train_index_list, test_index_list).
        Each tuple element is a list of n_rows*n_cols 1D array of sample indices.
    """
    if shuffle:
        indices_table = indices_table[np.random.permutation(indices_table.shape[0]), :]
        indices_table = indices_table[:, np.random.permutation(indices_table.shape[0])]
    if mask is None:
        mask = np.ones(indices_table.shape, dtype='bool')
    train_index_list = []
    test_index_list = []
    s0_index_list = None
    N1, N2 = indices_table.shape
    if k is not None:
        k1 = k
        k2 = k
    k1 = min(indices_table.shape[0], k1)
    k2 = min(indices_table.shape[1], k2)
    size1 = int(np.ceil(float(indices_table.shape[0])/k1))
    size2 = int(np.ceil(float(indices_table.shape[1])/k2))
    if method == 's0':
        s0_index_list = []
        for i in range(k1):
            for j in range(k2):
                test_index1 = np.r_[(size1*i):min(size1*(i + 1), N1)]
                train_index1 = np.setdiff1d(np.r_[:N1], test_index1)
                test_index2 = np.r_[(size2*j):min(size2*(j + 1), N2)]
                train_index2 = np.setdiff1d(np.r_[:N2], test_index2)

                train_index = np.ravel(indices_table[train_index1, :][:, train_index2])
                train_index_list.append(train_index[mask[train_index]])
                test_index = np.setdiff1d(np.ravel(indices_table), train_index)
                test_index_list.append(test_index[mask[test_index]])
                s0_index = np.ravel(indices_table[test_index1, :][:, test_index2])
                s0_index_list.append(s0_index[mask[s0_index]])
    elif method == 's1f':
        for i in range(k1):
            test_index1 = np.r_[(size1 * i):min(size1 * (i + 1), N1)]
            train_index1 = np.setdiff1d(np.r_[:N1], test_index1)

            train_index = np.ravel(indices_table[train_index1, :])
            train_index_list.append(train_index[mask[train_index]])
            test_index = np.ravel(indices_table[test_index1, :])
            test_index_list.append(test_index[mask[test_index]])
    elif method == 's1m':
        for j in range(k2):
            test_index2 = np.r_[(size2 * j):min(size2 * (j + 1), N2)]
            train_index2 = np.setdiff1d(np.r_[:N2], test_index2)

            train_index = np.ravel(indices_table[:, train_index2])
            train_index_list.append(train_index[mask[train_index]])
            test_index = np.ravel(indices_table[:, test_index2])
            test_index_list.append(test_index[mask[test_index]])
    elif method == 'block':
        for i in range(k1):
            for j in range(k2):
                test_index1 = np.r_[(size1 * i):min(size1 * (i + 1), N1)]
                train_index1 = np.setdiff1d(np.r_[:N1], test_index1)
                test_index2 = np.r_[(size2 * j):min(size2 * (j + 1), N2)]
                train_index2 = np.setdiff1d(np.r_[:N2], test_index2)

                train_index = np.ravel(indices_table[train_index1, :][:, train_index2])
                train_index_list.append(train_index[mask[train_index]])
                test_index = np.ravel(indices_table[test_index1, :][:, test_index2])
                test_index_list.append(test_index[mask[test_index]])
    else:
        raise ValueError('unknown cross-validation method: %s'%method)

    return train_index_list, test_index_list, s0_index_list
