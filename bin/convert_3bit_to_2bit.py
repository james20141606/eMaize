#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('convert_3bit_to_2bit')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert genotype from 3-bit code to 2-bit code. '
                                     'In 3-bit code, each genotype is encoded as 3 0/1 integers: AA->100, AB->010, BB->001. '
                                     'In 2-bit code, each genotype is encoded as 3 0/1 integers: AA->10, AB->11, BB->01.')
    parser.add_argument('--genotype-file', '-i', type=str, required=True,
                        help='3-bit code genotype file in binary format (int8)')
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='2-bit code genotype file in HDF5 format with dataset name "data"')
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import h5py

    logger.info('read phenotypes from {}'.format(args.phenotype_file))
    phenotypes = pd.read_table(args.phenotype_file)
    n_samples = phenotypes.shape[0]

    logger.info('total number of samples: {}'.format(n_samples))
    logger.info('read genotypes from {}'.format(args.genotype_file))
    with open(args.genotype_file, 'rb') as f:
        genotypes = np.frombuffer(f.read(), dtype='int8').reshape((-1, n_samples))
    n_genotypes = genotypes.shape[0]/3
    logger.info('number of genotypes: %d'%n_genotypes)
    logger.info('convert 2-bit genotype to 2-bit genotype')
    genotypes_2bit = np.empty((n_genotypes*2, n_samples), dtype='int8')
    genotypes_2bit[0::2] = genotypes[0::3]
    genotypes_2bit[0::2] += genotypes[1::3]
    genotypes_2bit[1::2] = genotypes[2::3]
    genotypes_2bit[1::2] += genotypes[1::3]
    del genotypes

    logger.info('save 2-bit genotypes to ' + args.output_file)
    prepare_output_file(args.output_file)
    f = h5py.File(args.output_file, 'w')
    f.create_dataset('data', data=genotypes_2bit)
    f.close()

