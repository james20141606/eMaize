#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_regression2')

def _get_session():
    """Modified the original get_session() function to change the ConfigProto variable
    """
    import tensorflow as tf
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if not _MANUAL_VAR_INIT:
        _initialize_variables()
    return session

def np_setdiff(a, b):
    values, counts = np.unique(np.concatenate((np.ravel(a), np.repeat(np.ravel(b), 2))), return_counts=True)
    return np.compress(counts == 1, values)

def data_generator_diagonal(X, y, indices=None, indices_na=None):
    row_order = np.arange(indices.shape[0])
    col_order = np.arange(indices.shape[1])
    while True:
        np.random.shuffle(row_order)
        np.random.shuffle(col_order)
        indices_rand = np.take(indices, row_order, axis=0)
        indices_rand = np.take(indices_rand, col_order, axis=1)
        for i in range(indices.shape[1]):
            ii = np.arange(indices.shape[0])
            jj = (ii + i)%indices.shape[1]
            indices_batch = np.ravel(indices_rand[ii, jj])
            indices_batch = np_setdiff(indices_batch, indices_na)
            X_batch = np.take(X, indices_batch, axis=0)
            y_batch = np.take(y, indices_batch, axis=0)
            yield X_batch, y_batch

def data_generator_random(X, y, indices, indices_na, batch_size=25):
    indices = np_setdiff(indices, indices_na)
    while True:
        indices_rand = np.copy(indices)
        np.random.shuffle(indices_rand)
        n_batches = indices.shape[0]/batch_size
        for i in range(n_batches):
            indices_batch = indices_rand[(i*batch_size):((i + 1)*batch_size)]
            X_batch = np.take(X, indices_batch, axis=0)
            y_batch = np.take(y, indices_batch, axis=0)
            yield X_batch, y_batch

def data_generator_byrow(X, y, indices, indices_na, batch_size=25):
    row_order = np.arange(indices.shape[0])
    while True:
        np.random.shuffle(row_order)
        indices_rand = np.ravel(np.take(indices, row_order, axis=0))
        indices_rand = np_setdiff(indices_rand, indices_na)
        n_batches = indices_rand.shape[0]/batch_size
        for i in range(n_batches):
            indices_batch = indices_rand[(i*batch_size):((i + 1)*batch_size)]
            X_batch = np.take(X, indices_batch, axis=0)
            y_batch = np.take(y, indices_batch, axis=0)
            yield X_batch, y_batch

def data_generator_bycol(X, y, indices, indices_na, batch_size=25):
    col_order = np.arange(indices.shape[1])
    while True:
        np.random.shuffle(col_order)
        indices_rand = np.ravel(np.take(indices, col_order, axis=1).T)
        indices_rand = np_setdiff(indices_rand, indices_na)
        n_batches = indices_rand.shape[0] / batch_size
        for i in range(n_batches):
            indices_batch = indices_rand[(i*batch_size):((i + 1)*batch_size)]
            X_batch = np.take(X, indices_batch, axis=0)
            y_batch = np.take(y, indices_batch, axis=0)
            yield X_batch, y_batch

def run_regression(args):
    import h5py
    import numpy as np
    from utils import read_hdf5_dataset, standardize_genotypes
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import pearsonr

    if args.gsm_file is not None:
        logger.info('read GSM file: ' + args.gsm_file)
        with h5py.File(args.gsm_file, 'r') as f:
            U = f['U'][:]
            S = f['S'][:]
            U = U*S[np.newaxis, :]
            U = U[:, S**2 > 0.5]
            U = standardize_genotypes(U)
            X = U
    else:
        logger.info('read genotype file: ' + args.genotype_file)
        X = read_hdf5_dataset(args.genotype_file)
        if args.transpose_x:
            logger.info('transpose X')
            X = X.T
        X = standardize_genotypes(X)
    logger.info('read phenotype file: ' + args.phenotype_file)
    y = read_hdf5_dataset(args.phenotype_file)
    logger.info('read training indices file: ' + args.train_index_file)
    train_index = read_hdf5_dataset(args.train_index_file)
    logger.info('read test indices file: ' + args.test_index_file)
    test_index = read_hdf5_dataset(args.test_index_file)

    if not os.path.exists(args.output_dir):
        logger.info('create output directory: ' + args.output_dir)
        os.makedirs(args.output_dir)
    logger.info('use model: ' + args.model_name)
    if args.model_name == 'mlp':
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        from keras import backend as K

        if K.backend() == 'tensorflow':
            # replace the original get_session() function
            keras.backend.tensorflow_backend.get_session.func_code = _get_session.func_code
        logger.info('build the model')
        model = Sequential()  # Feedforward
        model.add(Dense(500, input_dim=X.shape[1]))
        model.add(Activation('tanh'))
        model.add(Dense(100))
        model.add(Activation('tanh'))
        model.add(Dense(1))

        optimizer = keras.optimizers.RMSprop()
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        callbacks = [keras.callbacks.CSVLogger(os.path.join(args.output_dir, 'train_log.csv'))]
        logger.info('build the model')
        model.fit(X[train_index], y[train_index], 
            epochs=args.max_epochs,
            callbacks=callbacks)
        '''
        logger.info('save the model')
        model.save(os.path.join(args.output_dir, 'model'))'''
    else:
        logger.info('build the model')
        import dill as pickle
        if args.model_name == 'gpr':
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
            kernel = DotProduct(sigma_0=1.0)**4 + WhiteKernel()
            model = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        elif args.model_name == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1)
        logger.info('train the model')
        model.fit(X[train_index], y[train_index])
        '''
        logger.info('save the model')
        with open(os.path.join(args.output_dir, 'model'), 'wb') as fout:
            pickle.dump(model, fout)'''

    logger.info('test the model')
    y_pred = np.ravel(model.predict(X))

    logger.info('save predictions on the test set')
    fout = h5py.File(os.path.join(args.output_dir, 'predictions'), 'w')
    fout.create_dataset('y_true', data=y)
    fout.create_dataset('y_pred', data=y_pred)
    fout.close()

    for phase in ('train', 'test'):
        if phase == 'train':
            y_ = y[train_index]
            y_pred_ = y_pred[train_index]
        else:
            y_ = y[test_index]
            y_pred_ = y_pred[test_index]
        metrics = {}
        metrics['mean_squared_error'] = mean_squared_error(y_, y_pred_)
        metrics['r2_score'] = r2_score(y_, y_pred_)
        metrics['pearsonr'] = pearsonr(y_, y_pred_)[0]
        for metric_name, metric_value in metrics.items():
            logger.info('%s.%s = %f'%(phase, metric_name, metric_value))
        logger.info('save metrics')
        with open(os.path.join(args.output_dir, 'metrics.%s.txt'%phase), 'w') as fout:
            fout.write('%s\t%f\n'%(metric_name, metric_value))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--genotype-file', type=str,
                        help='input data in HDF5 format (e.g. filename:dataset)')
    g.add_argument('--gsm-file', type=str,
                        help='input data in HDF5 format (e.g. filename:dataset)')
    parser.add_argument('-y', '--phenotype-file', type=str, required=True,
                        help='input data in HDF5 format (e.g. filename:dataset)')                 
    parser.add_argument('--transpose-x', action='store_true',
                        help='transpose the rows and columns of X first')
    parser.add_argument('--train-index-file', type=str, required=True,
                        help='training sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--test-index-file', type=str, required=True,
                        help='test sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--model-name', type=str, default='mlp',
                        choices=('mlp', 'gpr', 'ridge'))
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    args = parser.parse_args()

    run_regression(args)

    

    