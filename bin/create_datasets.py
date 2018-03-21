#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('create_datasets')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Convert data for classification/regression')
    subparsers = main_parser.add_subparsers(dest='command')
    # command: merge
    parser = subparsers.add_parser('merge',
            help = 'merge multiple transformed feature files (HDF5 format) to a single matrix')
    parser.add_argument('-i', '--input-files', type=str, required=True, nargs='+',
                        help='transformed feature file (HDF5 format) with sample indices as dataset names')
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='an HDF5 file')
    # command: convert
    parser = subparsers.add_parser('convert',
                                    help='transformed feature file (HDF5 format) with sample indices as dataset names')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='transformed feature file (HDF5 format) with sample indices as dataset names')
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='an HDF5 file with two datasets: X, y')
    # command: normalize
    parser = subparsers.add_parser('normalize',
                                   help='normalize input features using sklearn.preprocessing.StandardScaler')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='transformed feature file (HDF5 format) produced by merge')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='normalized features in the same format as input file')
    parser.add_argument('--scaler-file', type=str, required=True,
                        help='scaler parameters in HDF5 format')
    # command: merge_parent
    parser = subparsers.add_parser('merge_parent',
                                   help='merge parent genotype (HDF5 format) to a single matrix')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='transformed feature file (HDF5 format) with sample indices as dataset names')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='an HDF5 file containing 2 datasets: X_female, X_male')

    args = main_parser.parse_args()

    import numpy as np
    import pandas as pd
    import h5py

    if args.command == 'merge':
        logger.info('read phenotypes from {}'.format(args.phenotype_file))
        phenotypes = pd.read_table(args.phenotype_file)
        n_samples_total = phenotypes.shape[0]
        X = {}
        for input_file in args.input_files:
            logger.info('read input file ' + input_file)
            fin = h5py.File(input_file, 'r')
            for dataset in fin.keys():
                X[int(dataset)] = fin[dataset][:]
            fin.close()
        datasets = sorted(X.keys())
        logger.info('merge all datasets into a single matrix')
        X = np.concatenate([X[dataset] for dataset in datasets]).reshape((len(datasets), -1))

        logger.info('save output file ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('X', data=X)
        g = fout.create_group('y')
        for trait in ['trait1', 'trait2', 'trait3']:
            g.create_dataset(trait, data=phenotypes[trait].values[datasets])
        fout.close()
    elif args.command == 'convert':
        logger.info('read phenotypes from {}'.format(args.phenotype_file))
        phenotypes = pd.read_table(args.phenotype_file)
        n_samples_total = phenotypes.shape[0]

        logger.info('read input file ' + args.input_file)
        fin = h5py.File(args.input_file, 'r')
        X = {}
        for dataset in fin.keys():
            X[int(dataset)] = fin[dataset][:]
        fin.close()

        datasets = sorted(X.keys())
        X = np.concatenate([X[dataset] for dataset in datasets]).reshape((len(datasets), -1))

        logger.info('save output file ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('X', data=X)
        for trait in ['trait1', 'trait2', 'trait3']:
            fout.create_dataset('y_' + trait, data=phenotypes[trait].values[datasets])
        fout.close()
    elif args.command == 'normalize':
        from sklearn.preprocessing import StandardScaler
        logger.info('read input file ' + args.input_file)
        fin = h5py.File(args.input_file, 'r')
        X = fin['X'][:]
        y = {}
        for trait in fin['y'].keys():
            y[trait] = fin['y/%s'%trait][:]
        fin.close()
        logger.info('normalize input features')
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(X)

        logger.info('save normalized features to ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('X', data=X)
        g = fout.create_group('y')
        for trait in y.keys():
            g.create_dataset(trait, data=y[trait])
        fout.close()

        logger.info('save normalization parameters to ' + args.scaler_file)
        prepare_output_file(args.scaler_file)
        fout = h5py.File(args.scaler_file, 'w')
        fout.create_dataset('scale_', data=scaler.scale_)
        fout.create_dataset('mean_', data=scaler.mean_)
        fout.close()
    elif args.command == 'merge_parent':
        def read_parent_genotypes(filename):
            genotypes_female = {}
            genotypes_male = {}
            f = h5py.File(filename, 'r')
            for parent in f.keys():
                if parent.startswith('f'):
                    number = int(parent[1:])
                    genotypes_female[number] = f[parent][:]
                elif parent.startswith('m'):
                    number = int(parent[1:])
                    genotypes_male[number] = f[parent][:]
            f.close()
            names_female = ['f%d'%i for i in range(len(genotypes_female))]
            X_female = np.concatenate([genotypes_female[i + 1] for i in range(len(genotypes_female))]).reshape(
                (len(genotypes_female), -1))
            names_male = ['m%d'%i for i in range(len(genotypes_male))]
            X_male = np.concatenate([genotypes_male[i + 1] for i in range(len(genotypes_male))]).reshape(
                (len(genotypes_male), -1))
            return X_female, X_male, names_female, names_male
        logger.info('read parent genotypes file: ' + args.input_file)
        X_female, X_male, names_female, names_male = read_parent_genotypes(args.input_file)
        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        fout.create_dataset('X_female', data=X_female)
        fout.create_dataset('X_male', data=X_male)
        fout.create_dataset('names_female', data=np.asarray(names_female))
        fout.create_dataset('names_male', data=np.asarray(names_male))
        fout.close()

