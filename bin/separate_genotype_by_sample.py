#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('separate_genotype_by_sample')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype-dir', '-i', type=str, required=True)
    parser.add_argument('--format', type=str, required=True, choices=('binary', 'hdf5'),
                        help='input file format')
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('-s', '--sample-start', type=int, required=True)
    parser.add_argument('-e', '--sample-end', type=int, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import h5py

    logger.info('read phenotypes from {}'.format(args.phenotype_file))
    phenotypes = pd.read_table(args.phenotype_file)
    n_samples_total = phenotypes.shape[0]
    phenotypes.index = phenotypes['id']
    #logger.info('phenotypes: %s' + str(phenotypes.head(3)))
    if args.sample_start > n_samples_total:
        logger.info('exit because sample_start is greater than total number of samples: %d'%n_samples_total)
        sys.exit(0)
    if args.sample_end > n_samples_total:
        logger.info('change sample_end to total number of samples: %d'%n_samples_total)
        args.sample_end = n_samples_total

    logger.info('read genotypes of sample from %d to %d'%(args.sample_start, args.sample_end))
    logger.info('genotype file format is %s'%args.format)
    genotypes_by_sample = []
    for chrom in ['chr%d'%i for i in range(1, 11)]:
        genotype_file = os.path.join(args.genotype_dir, chrom)
        logger.info('read genotypes from ' + genotype_file)
        if args.format == 'binary':
            with open(genotype_file, 'rb') as f:
                genotypes = np.frombuffer(f.read(), dtype='int8').reshape((-1, n_samples_total))
        elif args.format == 'hdf5':
            fin = h5py.File(genotype_file, 'r')
            genotypes = fin[fin.keys()[0]][:]
            fin.close()
        else:
            raise ValueError('unknown genotype file format: %s'%args.format)
        logger.info('take subset from genotype file: ' + genotype_file)
        genotypes_by_sample.append(np.take(genotypes, np.arange(args.sample_start, args.sample_end), axis=1).T)
        del genotypes
    logger.info('combine genotypes')
    genotypes_by_sample = np.concatenate(genotypes_by_sample, axis=1)

    sample_file = os.path.join(args.output_dir, '%d-%d'%(args.sample_start, args.sample_end))
    logger.info('save genotypes to ' + sample_file)
    prepare_output_file(sample_file)
    f = h5py.File(sample_file, 'w')
    for sample_id in range(args.sample_start, args.sample_end):
        f.create_dataset('%d'%sample_id, data=genotypes_by_sample[sample_id - args.sample_start])
    f.close()
