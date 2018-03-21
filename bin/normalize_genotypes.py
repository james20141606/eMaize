#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def convert_2bit_to_minor(args):
    from utils import read_hdf5_dataset, prepare_output_file
    import numpy as np
    import h5py
    import numba

    @numba.jit(nopython=True)
    def _2bit_to_minor(X_2bit, X_minor):
        n_snps = X_minor.shape[0]
        n_samples = X_minor.shape[1]
        max_freq = n_samples
        for i in range(n_snps):
            freq = 0
            for j in range(n_samples):
                count = X_2bit[i, 1, j] - X_2bit[i, 0, j] + 1
                freq += count
                X_minor[i, j] = count
            if freq > n_samples:
                for j in range(n_samples):
                    X_minor[i, j] = 2 - X_minor[i, j]

    logger.info('read input file: ' + args.input_file)
    X_2bit, dataset = read_hdf5_dataset(args.input_file, return_name=True)
    n_snps, n_samples = X_2bit.shape
    n_snps /= 2
    logger.info('number of SNPs: %d, number of samples: %d'%(n_snps, n_samples))
    X_2bit = X_2bit.reshape((n_snps, 2, n_samples))
    logger.info('convert from 2bit code to minor copy numbers')
    # assume that the second allele in the 2bit representation is the minor allele
    # 10 -> 0, 11 -> 1, 01 -> 2
    #X_minor = np.einsum('ijk,j->ik', X_2bit, np.array([-1, 1])) + 1
    # swap the two alleles to make sure that the number represents a minor allele
    #allele_freq = np.expand_dims(np.sum(X_2bit[:, 1, :], axis=1), axis=1)
    #X_minor = np.where(allele_freq <= n_samples/2, X_minor, 2 - X_minor)
    X_minor = np.empty((n_snps, n_samples), dtype='int8')
    _2bit_to_minor(X_2bit, X_minor)
    logger.info('save minor allele copy numbers to output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset(dataset, data=X_minor)

def normalize_genotypes(args):
    from utils import read_hdf5_dataset, prepare_output_file
    import numpy as np
    import h5py

    logger.info('read input file: ' + args.input_file)
    X, dataset = read_hdf5_dataset(args.input_file, return_name=True)
    n_snps, n_samples = X.shape
    # allele frequencies
    p = X.sum(axis=1).astype('float32')/n_samples
    multiplier = 1.0/np.sqrt(2.0*p*(1.0 - p))
    multiplier = multiplier.astype('float32')
    logger.info('save mean and multipliers to output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('mean', data=p)
        f.create_dataset('multiplier', data=multiplier)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Normalize genotypes to zero mean and unit variance')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('2bit_to_minor',
                                   help='convert 2-bit representation of genotype to minor allele copy numbers (0, 1, 2)')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='2-bit code genotype file in HDF5 format (by chromosome)')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format')

    parser = subparsers.add_parser('normalize_genotypes',
                                   help='calculate mean and multipliers (1/std) for genotypes')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='2-bit code genotype file in HDF5 format (by chromosome)')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format with two datasets: "mean" and "multiplier"')

    args = main_parser.parse_args()

    logger = logging.getLogger('normalize_genotypes.' + args.command)
    if args.command == '2bit_to_minor':
        convert_2bit_to_minor(args)
    elif args.command == 'normalize_genotypes':
        normalize_genotypes(args)