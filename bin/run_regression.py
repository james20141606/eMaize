#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('random_projection')

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input data in HDF5 format (e.g. filename:dataset)')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='phenotype for each sample')
    parser.add_argument('-x', '--xname', type=str, default='X',
                        help='dataset name of X')
    parser.add_argument('--transpose-x', action='store_true',
                        help='transpose the rows and columns of X first')
    parser.add_argument('--standardize-x', action='store_true',
                        help='transform features of X to zero means and unit variances')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-type', type=str, default='diagonal',
                        choices=('diagonal','random', 'byrow', 'bycol'))
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--model-name', type=str, default='mlp',
                        choices=('mlp', 'gpr', 'ridge'))
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    args = parser.parse_args()

    logger.info('read input data: ' + args.input_file)
    import h5py
    fin = h5py.File(args.input_file, 'r')
    X = fin[args.xname][:]
    fin.close()
    if args.transpose_x:
        logger.info('transpose X')
        X = X.T

    import pandas as pd
    import numpy as np
    from utils import generate_parent_table, read_hdf5_dataset

    logger.info('read phenotypes: ' + args.phenotype_file)
    phenotypes = pd.read_table(args.phenotype_file)
    parent_table = generate_parent_table(args.phenotype_file)
    y = phenotypes['trait1'].values

    pedigree = phenotypes['pedigree'].str.split('_', expand=True)
    pedigree.columns = ['f', 'X', 'm']
    male_ids = ['m%d' % i for i in range(1, 31)]
    female_ids = ['f%d' % i for i in range(1, 208)]
    phenotypes = pd.concat([phenotypes, pedigree], axis=1)
    phenotypes['number'] = np.arange(phenotypes.shape[0])
    parent_table = phenotypes.pivot_table(values='number', index=['m'], columns=['f'], dropna=False)
    male_ids = ['m%d' % i for i in range(1, parent_table.shape[0] + 1)]
    female_ids = ['f%d' % i for i in range(1, parent_table.shape[1] + 1)]
    parent_table = parent_table.loc[male_ids, female_ids]
    parent_table = np.copy(parent_table.iloc[:25, :192].values)
    samples_na = np.nonzero(np.isnan(phenotypes['trait1']).values)

    if args.bootstrap:
        rows_train = np.random.choice(parent_table.shape[0], size=20, replace=False)
        cols_train = np.random.choice(parent_table.shape[1], size=160, replace=False)
        parent_table_train = np.take(np.take(parent_table, rows_train, axis=0), cols_train, axis=1)
        indices_test = np_setdiff(parent_table, parent_table_train)
        indices_test = np_setdiff(indices_test, samples_na)
        X_test = X[indices_test]
        y_test = y[indices_test]
        validation_data = (X_test[:500], y_test[:500])
        logger.info('number of test samples: %d' % indices_test.shape[0])
    else:
        rows_train = np.arange(parent_table.shape[0])
        cols_train = np.arange(parent_table.shape[1])
        parent_table_train = parent_table
        indices_test = np.nonzero((phenotypes['type'] == 'test').values)[0]
        X_test = X[indices_test]
        y_test = None
        validation_data = None

    logger.info('number of training samples: %d'%np_setdiff(parent_table_train, samples_na).shape[0])

    if not os.path.isdir(args.output_dir):
        logger.info('create output directory: ' + args.output_dir)
        os.makedirs(args.output_dir)
    np.savetxt(os.path.join(args.output_dir, 'parent_table.txt'),
               parent_table, fmt='%d', delimiter='\t')
    sampling_table = np.sum(np.expand_dims(parent_table, axis=-1) == parent_table_train.reshape((1, 1, -1)), axis=2)
    np.savetxt(os.path.join(args.output_dir, 'sampling_table.txt'),
               sampling_table, fmt='%d', delimiter='\t')
    np.savetxt(os.path.join(args.output_dir, 'male_parents_train.txt'),
               rows_train, fmt='%d')
    np.savetxt(os.path.join(args.output_dir, 'female_parents_train.txt'),
               cols_train, fmt='%d')
    np.savetxt(os.path.join(args.output_dir, 'indices_train.txt'),
               np_setdiff(parent_table_train, samples_na), fmt='%d')
    if args.bootstrap:
        np.savetxt(os.path.join(args.output_dir, 'indices_test.txt'),
                   indices_test, fmt='%d')

    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import pearsonr

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
        model.add(Dense(1000, input_dim=X.shape[1]))
        model.add(Activation('tanh'))
        model.add(Dense(200))
        model.add(Activation('tanh'))
        model.add(Dense(1))

        optimizer = keras.optimizers.RMSprop()
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        if args.batch_type == 'diagonal':
            generator = data_generator_diagonal(X, y, parent_table_train, samples_na)
            steps_per_epoch = parent_table_train.shape[1]
        elif args.batch_type == 'random':
            generator = data_generator_random(X, y, parent_table_train, samples_na, batch_size=25)
            steps_per_epoch = np_setdiff(parent_table_train, samples_na).shape[0]/25
        elif args.batch_type == 'byrow':
            generator = data_generator_byrow(X, y, parent_table_train, samples_na, batch_size=25)
            steps_per_epoch = np_setdiff(parent_table_train, samples_na).shape[0]/25
        elif args.batch_type == 'bycol':
            generator = data_generator_bycol(X, y, parent_table_train, samples_na, batch_size=25)
            steps_per_epoch = np_setdiff(parent_table_train, samples_na).shape[0]/25
        else:
            raise ValueError('unknown batch type: %s' % args.batch_type)
        callbacks = [keras.callbacks.CSVLogger(os.path.join(args.output_dir, 'train_log.csv'))]
        logger.info('build the model')
        model.fit_generator(generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=args.max_epochs,
                            callbacks=callbacks)

        logger.info('save the model')
        model.save(os.path.join(args.output_dir, 'model'))
    else:
        logger.info('build the model')
        import dill as pickle
        if args.model_name == 'gpr':
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
            kernel = DotProduct() + WhiteKernel()
            model = GaussianProcessRegressor(kernel=kernel)
        elif args.model_name == 'ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1000)
        indices_train = np_setdiff(parent_table_train, samples_na)
        logger.info('train the model')
        model.fit(X[indices_train], y[indices_train])
        logger.info('save the model')
        with open(os.path.join(args.output_dir, 'model'), 'wb') as fout:
            pickle.dump(model, fout)

    if args.bootstrap:
        logger.info('test the model')
        y_pred = np.ravel(model.predict(X_test))

        logger.info('save predictions on the test set')
        fout = h5py.File(os.path.join(args.output_dir, 'predictions'), 'w')
        fout.create_dataset('y_true', data=y_test)
        fout.create_dataset('y_pred', data=y_pred)
        fout.close()

        metrics = {}
        metrics['mean_squared_error'] = mean_squared_error(y_test, y_pred)
        metrics['r2_score'] = r2_score(y_test, y_pred)
        metrics['pearsonr'] = pearsonr(y_test, y_pred)[0]
        for metric_name, metric_value in metrics.items():
            logger.info('%s = %f'%(metric_name, metric_value))
        logger.info('save metrics')
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as fout:
            fout.write('%s\t%f\n'%(metric_name, metric_value))
    else:
        logger.info('predict test data')
        y_pred = np.ravel(model.predict(X_test))
        results = pd.DataFrame({'number': indices_test,
                                'id': phenotypes['id'][indices_test],
                                'prediction': y_pred})
        results.to_csv(os.path.join(args.output_dir, 'predictions.txt'),
                       sep='\t', quoting=False)