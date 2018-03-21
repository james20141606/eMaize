#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('infer_parent_genotypes')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

def generate_parent_table(phenotype_file):
    phenotypes = pd.read_table(phenotype_file)
    pedigree = phenotypes['pedigree'].str.split('_', expand=True)
    pedigree.columns = ['f', 'X', 'm']
    phenotypes = pd.concat([phenotypes, pedigree], axis=1)
    phenotypes['number'] = np.arange(phenotypes.shape[0])
    parent_table = phenotypes.pivot_table(values='number', index=['m'], columns=['f'], dropna=False)
    male_ids = ['m%d' % i for i in range(1, parent_table.shape[0] + 1)]
    female_ids = ['f%d' % i for i in range(1, parent_table.shape[1] + 1)]
    parent_table = parent_table.loc[male_ids, female_ids]
    return parent_table

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Infer parent genotypes from genotypes of descents')
    subparsers = main_parser.add_subparsers(dest='command')
    # command: infer
    parser = subparsers.add_parser('infer',
                                   help='infer parent genotypes for each genotype file')
    parser.add_argument('--genotype-file', '-i', type=str, required=True,
                        help='2-bit code genotype file in HDF5 format')
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('--counts', action='store_true',
                        help='save counts of each genotype')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='2-bit code genotype file in HDF5 format with dataset name "data"')

    # command: merge
    parser = subparsers.add_parser('merge',
                                   help='merge multiple genotypes file into a single file')
    parser.add_argument('--genotype-files', '-i', type=str, nargs='+', required=True,
                        help='2-bit code genotype files in HDF5 format')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='2-bit code genotype file in HDF5 format with dataset name "data"')
    args = main_parser.parse_args()

    import h5py
    import pandas as pd
    import numpy as np

    if args.command == 'infer':
        logger.info('read phenotypes from {}'.format(args.phenotype_file))
        parent_table = generate_parent_table(args.phenotype_file)

        logger.info('read genotypes from ' + args.genotype_file)
        f = h5py.File(args.genotype_file, 'r')
        genotypes = f['data'][:]
        f.close()

        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        n_females = parent_table.shape[0]
        n_males = parent_table.shape[1]
        for parent in parent_table.columns:
            logger.info('infer genotypes for female parent %s'%parent)
            genotypes_parent = genotypes[:, parent_table.loc[:, parent].values].sum(axis=1)
            if not args.counts:
                genotypes_parent = (genotypes_parent >= (n_females - 3)).astype('int8')
            fout.create_dataset(parent, data=genotypes_parent)
        for parent in parent_table.index:
            logger.info('infer genotypes for male parent %s' % parent)
            genotypes_parent = genotypes[:, parent_table.loc[parent, :].values].sum(axis=1)
            if not args.counts:
                genotypes_parent = (genotypes_parent >= (n_males - 3)).astype('int8')
            fout.create_dataset(parent, data=genotypes_parent)
        fout.close()
    elif args.command == 'merge':
        genotypes = {}
        for filename in args.genotype_files:
            logger.info('read genotype file: ' + filename)
            fin = h5py.File(filename, 'r')
            for parent in fin.keys():
                if parent not in genotypes:
                    genotypes[parent] = []
                genotypes[parent].append(fin[parent][:])
            fin.close()
        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        for parent in genotypes:
            fout.create_dataset(parent, data=np.concatenate(genotypes[parent]))
        fout.close()

