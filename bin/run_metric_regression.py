#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_metric_regression')

def test(args):
    from metric_regressor import MetricRegressor
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import Ridge

    logger.info('generate random data')
    N = args.n_samples
    X = np.random.normal(size=(N*2, args.n_features))
    w = np.random.normal(scale=np.sqrt(1.0/args.n_features), size=(args.n_features, 1))
    e = np.random.normal(scale=0.1, size=(N*2, 1))
    y = X.dot(w) + e
    X_train, X_test = X[:N], X[N:]
    y_train, y_test = y[:N], y[N:]
    logger.info('y_mean=%f, y_var=%f'%(y.mean(), y.std()**2))

    def random_pair_generator(X, y, size=4):
        N = X.shape[0]
        while True:
            indices = np.random.choice(N, size=size*2)
            train_index, test_index = indices[:size], indices[size:]
            yield X[train_index], X[test_index], y[train_index], y[test_index]

    logger.info('run metric regression')
    model = MetricRegressor(input_dim=args.n_features, hidden_dim=args.n_hidden, alpha=args.alpha,
                            sparse_rate=args.sparse_rate)
    model.fit(X_train, y_train, data_generator=random_pair_generator(X, y, size=5),
              lr=args.lr, max_iter=args.max_iter, n_batches=args.n_splits)
    y_pred = model.predict(X_test)
    logger.info('training MSE using metric regression: %f'%model.mses_[-1])
    logger.info('test MSE using metric regression: %f'%mean_squared_error(y_test, y_pred))

    logger.info('run linear regression')
    model_ridge = Ridge(args.alpha)
    model_ridge.fit(X_train, y_train)
    y_pred_ridge = model_ridge.predict(X_test)
    logger.info('test MSE using ridge regression: %f'%mean_squared_error(y_test, y_pred_ridge))

def run_metric_regressor(args):
    from utils import read_hdf5_dataset
    import h5py
    from metric_regressor import MetricRegressor
    import dill as pickle
    import numpy as np
    from utils import read_hdf5_single, cv_split_emaize, standardize_genotypes, get_indices_table

    logger.info('read genotype file: ' + args.genotype_file)
    X = read_hdf5_dataset(args.genotype_file)
    if args.transpose_genotype:
        X = X.T

    X = standardize_genotypes(X)
    logger.info('read GSM file: ' + args.gsm_file)
    with h5py.File(args.gsm_file, 'r') as f:
        U = f['U'][:]
        S = f['S'][:]
        U = U[:, S ** 2 > 0.5]
        U = standardize_genotypes(U)
    logger.info('read phenotype file: ' + args.phenotype_file)
    y = read_hdf5_dataset(args.phenotype_file)
    logger.info('read parent table file: ' + args.parent_table_file)
    parent_table = read_hdf5_single(args.parent_table_file)
    logger.info('read training indices file: ' + args.train_index_file)
    train_index = read_hdf5_dataset(args.train_index_file)
    logger.info('read test indices file: ' + args.train_index_file)
    test_index = read_hdf5_dataset(args.test_index_file)

    indices_table, mask = get_indices_table(train_index, parent_table)
    if args.cv_type == 's1f':
        train_index_list, test_index_list, s0_index_list = cv_split_emaize(indices_table, mask,
                                                                           k=parent_table.shape[0], method='s1f')
    elif args.cv_type == 's0':
        train_index_list, test_index_list, s0_index_list = cv_split_emaize(indices_table, mask,
                                                                           k1=parent_table.shape[0] / 5,
                                                                           k2=parent_table.shape[1] / 5,
                                                                           method='s0')
    elif args.cv_type == 's1m':
        train_index_list, test_index_list, s0_index_list = cv_split_emaize(indices_table, mask,
                                                                           k=parent_table.shape[1], method='s1m')
    else:
        raise ValueError('unkown cross-validation type: %s'%args.cv_type)

    logger.info('%d rows and %d columns in the indices table' % (indices_table.shape[0], indices_table.shape[1]))
    logger.info('number of cross-validation folds %d' % len(train_index_list))
    logger.info('number principal components to use: %d' % U.shape[1])

    X = np.concatenate([X, U], axis=1)
    y = y.reshape((-1, 1))
    X_train, y_train = X[train_index], y[train_index]

    def cv_generator(batch_size=5):
        while True:
            for i in range(len(train_index_list)):
                train_index = train_index_list[i]
                if args.cv_type == 's0':
                    test_index = test_index_list[i][s0_index_list[i]]
                else:
                    test_index = test_index_list[i]
                if (len(train_index) < batch_size) or (len(test_index) < batch_size):
                    continue
                train_index = np.random.choice(train_index, size=batch_size, replace=False)
                test_index = np.random.choice(test_index, size=batch_size, replace=False)
                yield (X_train[train_index], X_train[test_index],
                       y_train[train_index], y_train[test_index])

    model = MetricRegressor(input_dim=X.shape[1], hidden_dim=args.n_hidden, alpha=args.alpha,
                            sparse_rate=args.sparse_rate, kernel=args.kernel)
    model.fit(X_train, y_train, data_generator=cv_generator(batch_size=args.batch_size),
              lr=args.lr, max_iter=args.max_iter, n_batches=len(train_index_list))
    y_pred = model.predict(X)

    logger.info('save results to output directory: ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_file = os.path.join(args.output_dir, 'model')
    #with open(model_file, 'wb') as f:
    #    pickle.dump(model, f)
    model.save(model_file)
    pred_file = os.path.join(args.output_dir, 'predictions')
    with h5py.File(pred_file, 'w') as f:
        f.create_dataset('y_true', data=y)
        f.create_dataset('y_pred', data=y_pred)
        f.create_dataset('mses', data=model.mses_)
        f.create_dataset('velocities', data=model.velocities_)
        f.create_dataset('mse_grads', data=model.mse_grads_)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Commands for metric regression')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('test')
    parser.add_argument('-n', '--n-samples', type=int, default=100)
    parser.add_argument('-p', '--n-features', type=int, default=10)
    parser.add_argument('-q', '--n-hidden', type=int, default=10)
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='L2 regularization factor')
    parser.add_argument('-k', '--n-splits', type=int, default=3)
    parser.add_argument('-r', '--sparse-rate', type=float, default=0.5)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)

    parser = subparsers.add_parser('metric_regressor')
    parser.add_argument('-x', '--genotype-file', type=str, required=True,
                        help='genotype file (rows are samples and columns are SNPs')
    parser.add_argument('--transpose-genotype', action='store_true',
                        help='transpose the genotype matrix')
    parser.add_argument('-y', '--phenotype-file', type=str, required=True,
                        help='phenotype file. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--gsm-file', type=str, required=True,
                        help='genetic similarity matrix (GSM) file')
    parser.add_argument('--cv-type', type=str, default='s0',
                        choices=('s0', 's1f', 's1m'))
    parser.add_argument('-q', '--n-hidden', type=int, default=10)
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='L2 regularization factor')
    parser.add_argument('-k', '--batch-size', type=int, default=5)
    parser.add_argument('-r', '--sparse-rate', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='linear_ard',
                        choices=('linear_ard', 'projection'))
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--parent-table-file', type=str, required=True,
                        help='parent table in HDF5 format')
    parser.add_argument('--train-index-file', type=str, required=True,
                        help='training sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--test-index-file', type=str, required=True,
                        help='test sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='output directory')

    args = main_parser.parse_args()
    logger = logging.getLogger('run_metric_regression.' + args.command)
    if args.command == 'test':
        test(args)
    elif args.command == 'metric_regressor':
        run_metric_regressor(args)
