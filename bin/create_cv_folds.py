#! /usr/bin/env python
import argparse, os, sys, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('create_cv_folds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True,
        help='phenotype file in text format (pheno_emaize.txt)')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='output file in HDF5 format')
    parser.add_argument('-m', '--method', type=str, default='random',
                        choices=('random', 'by_female', 'by_male', 'cross'),
                        help='type of cross-validation')
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--k-female', type=int, default=5)
    parser.add_argument('--k-male', type=int, default=5)
    parser.add_argument('--max-size', type=int, help='maximum number of index sets to generate')
    parser.add_argument('-s', '--seed', type=int, help='random seed for numpy')
    args = parser.parse_args()

    from utils import prepare_output_file
    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd
    from utils import generate_parent_table

    if args.seed is not None:
        logger.info('set random seed for numpy: %d'%args.seed)
        np.random.seed(args.seed)

    logger.info('read phenotype file: ' + args.input_file)
    phenotypes = pd.read_table(args.input_file)
    parent_table = generate_parent_table(args.input_file)
    is_training = (phenotypes['type'] == 'training').values
    is_training_table = np.take(is_training, parent_table)
    ind_female_training = np.nonzero(~np.all(is_training_table, axis=0))[0]
    ind_male_training = np.nonzero(~np.all(is_training_table, axis=1))[0]
    parent_table_training = parent_table[ind_male_training][:, ind_female_training]

    train_index = []
    test_index = []
    if args.method == 'random':
        ind_training = np.nonzero((phenotypes['type'] == 'training').values)[0]
        kfold = KFold(args.k, shuffle=True)
        for train_index_i, test_index_i in kfold.split(ind_training):
            train_index.append(ind_training[train_index_i])
            test_index.append(ind_training[test_index_i])
    elif args.method == 'by_female':
        kfold = KFold(args.k, shuffle=True)
        for train_index_i, test_index_i in kfold.split(ind_female_training):
            train_index_i = ind_female_training[train_index_i]
            test_index_i = ind_female_training[test_index_i]
            parent_table_train = np.ravel(parent_table[ind_male_training][:, train_index_i])
            train_index.append(parent_table_train[is_training[parent_table_train]])
            parent_table_test = np.ravel(parent_table[ind_male_training][:, test_index_i])
            test_index.append(parent_table_test[is_training[parent_table_test]])
    elif args.method == 'by_male':
        kfold = KFold(args.k, shuffle=True)
        for train_index_i, test_index_i in kfold.split(ind_male_training):
            train_index_i = ind_male_training[train_index_i]
            test_index_i = ind_male_training[test_index_i]
            parent_table_train = np.ravel(parent_table[:, ind_female_training][train_index_i])
            train_index.append(parent_table_train[is_training[parent_table_train]])
            parent_table_test = np.ravel(parent_table[:, ind_female_training][test_index_i])
            test_index.append(parent_table_test[is_training[parent_table_test]])
    elif args.method == 'cross':
        kfold_female = KFold(args.k_female, shuffle=True)
        kfold_male = KFold(args.k_male, shuffle=True)
        for train_index_female, _ in kfold_female.split(ind_female_training):
            train_index_female = ind_female_training[train_index_female]
            for train_index_male, _ in kfold_male.split(ind_male_training):
                train_index_male = ind_male_training[train_index_male]
                parent_table_train= np.ravel(parent_table[:, train_index_female][train_index_male])
                train_index.append(parent_table_train[is_training[parent_table_train]])
                parent_table_test = np.setdiff1d(np.ravel(parent_table_training), parent_table_train)
                test_index.append(parent_table_test[is_training[parent_table_test]])

    import h5py
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as fout:
        if args.max_size is not None:
            if len(train_index) > args.max_size:
                logger.info('randomly sample %d sets from %s sets'%(args.max_size, len(train_index)))
                sel = np.random.choice(len(train_index), size=args.max_size, replace=False)
                train_index = [train_index[i] for i in sel]
                test_index = [test_index[i] for i in sel]
        for i in range(len(train_index)):
            g = fout.create_group(str(i))
            g.create_dataset('train', data=train_index[i])
            g.create_dataset('test', data=test_index[i])