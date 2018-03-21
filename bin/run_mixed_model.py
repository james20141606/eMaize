#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_mixed_model')

def fast_linregress_cv(x, y, train_masks, test_masks, eps=1e-20):
    """

    :param X: 1D array of shape (n_samples,)
    :param y: 1D array of shape (n_samples,)
    :param train_masks: 2D array of shape (n_cv, n_samples)
    :param test_masks: 2D array of shape (n_cv, n_samples)
    :param eps:
    :return: mse_train, mse_test: 1D arrays of shape (n_cv,)
    """
    import numpy as np
    y = y[np.newaxis, :]
    x = x[np.newaxis, :]

    N_train = np.sum(train_masks, axis=1)[:, np.newaxis]
    xm = np.nansum(x*train_masks, axis=1)[:, np.newaxis]/N_train
    ym = np.nansum(y*train_masks, axis=1)[:, np.newaxis]/N_train
    x_ = x - xm
    y_ = y - ym
    ssx = np.nansum((x_**2)*train_masks, axis=1)[:, np.newaxis]
    ssxy = np.nansum((x_*y_)*train_masks, axis=1)[:, np.newaxis]
    w = ssxy/(ssx + eps)
    b = ym - w*xm
    y_estim = w*x + b

    N_test = np.sum(test_masks, axis=1)[:, np.newaxis]
    sse = (y - y_estim)**2
    mse_train = np.nansum(sse*train_masks, axis=1)[:, np.newaxis]/N_train
    mse_test = np.nansum(sse*test_masks, axis=1)[:, np.newaxis]/N_test

    return np.ravel(mse_train), np.ravel(mse_test)

def single_model(args):
    import h5py
    import pandas as pd
    import numpy as np
    import dill as pickle
    from utils import read_hdf5_dataset, prepare_output_file, read_hdf5_single
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm

    logger.info('read phenotypes from file: ' + args.phenotype_file)
    #phenotypes = pd.read_table(args.phenotype_file)
    phenotypes = read_hdf5_dataset(args.phenotype_file)
    logger.info('read genotypes from file: ' + args.genotype_file)
    X = read_hdf5_dataset(args.genotype_file)
    if args.transpose_x:
        logger.info('transpose X')
        X = X.T
    y = phenotypes
    if args.feature_indices_file:
        logger.info('read feature indices from: ' + args.feature_indices_file)
        feature_indices = read_hdf5_dataset(args.feature_indices_file)
        X = np.take(X, feature_indices, axis=1)
    if args.normalize_x:
        logger.info('normalize X')
        X = StandardScaler().fit_transform(X)
    if args.sample_indices_file:
        logger.info('read sample indices from: ' + args.sample_indices_file)
        sample_indices = read_hdf5_dataset(args.sample_indices_file)
    else:
        sample_indices = np.nonzero(~np.isnan(phenotypes))[0]
    X_train = X[sample_indices]
    y_train = y[sample_indices]
    logger.info('read parent table from file: ' + args.parent_table_file)
    parent_table = read_hdf5_single(args.parent_table_file)

    logger.info('use model ' + args.model_name)
    logger.info('X.shape = %s, y.shape = %s'%(repr(X.shape), repr(y.shape)))
    if args.model_name == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=10000)
        model.fit(X_train, y_train)
        y_pred = np.ravel(model.predict(X))
        y_pred_train = y_pred[sample_indices]
    elif args.model_name == 'ridge_cv':
        from sklearn.linear_model import Ridge
        alphas = 10.0**np.arange(1, 6)
        train_masks, test_masks = generate_cv_masks(sample_indices, parent_table, k_female=5, k_male=5)
        cv_metrics = {}
        cv_metrics['mse'] = np.zeros((len(alphas), train_masks.shape[0]))
        cv_metrics['r2'] = np.zeros((len(alphas), train_masks.shape[0]))
        pbar = tqdm(total=len(alphas)*train_masks.shape[0])
        for i, alpha in enumerate(alphas):
            for j in range(train_masks.shape[0]):
                model = Ridge(alpha=alpha)
                model.fit(X[train_masks[j]], y[train_masks[j]])
                y_pred = model.predict(X[test_masks[j]])
                cv_metrics['mse'][i, j] = mean_squared_error(y[test_masks[j]], y_pred)
                cv_metrics['r2'][i, j] = r2_score(y[test_masks[j]], y_pred)
                pbar.update(1)
        pbar.close()
        best_alpha = alphas[cv_metrics['r2'].mean(axis=1).argmax()]
        logger.info('optmized alpha = %f'%best_alpha)
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        y_pred = np.ravel(model.predict(X))
        y_pred_train = y_pred[sample_indices]
    elif args.model_name == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
        kernel = RBF() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X_train, y_train)
        logger.info('kernel params: %s'%repr(model.get_params()))
        y_pred_train = np.ravel(model.predict(X_train))
        y_pred = np.ravel(model.predict(X))
    elif args.model_name == 'gpy':
        from GPy.kern import Linear
        from GPy.models import GPRegression
        kernel = Linear(input_dim=2, name='linear')
        model = GPRegression(X_train, y_train, kernel=kernel)
        model.optimize()

    else:
        raise ValueError('unknown model name: ' + args.model_name)

    logger.info('r2 score = %f'%r2_score(y_train, y_pred_train))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_file = os.path.join(args.output_dir, 'model')
    logger.info('save model file: ' + model_file)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    pred_file = os.path.join(args.output_dir, 'predictions')
    logger.info('save predictions to file: ' + pred_file)
    with h5py.File(pred_file, 'w') as f:
        if args.output_residuals:
            f.create_dataset('residual', data=(y - y_pred))
        f.create_dataset('y_true', data=y)
        f.create_dataset('y_pred', data=y_pred)
        f.create_dataset('y_pred_train', data=y_pred_train)
        f.create_dataset('indices_train', data=sample_indices)
        if args.model_name == 'ridge_cv':
            f.create_dataset('alpha', data=alphas)
            g = f.create_group('cv_metrics')
            for key in cv_metrics.keys():
                g.create_dataset(key, data=cv_metrics[key])

def evaluate(args):
    import h5py
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import pearsonr
    from utils import prepare_output_file, read_hdf5_dataset

    logger.info('read prediction file: ' + args.input_file)
    with h5py.File(args.input_file, 'r') as f:
        y_true = f['y_true'][:]
        y_pred = f['y_pred'][:]
    logger.info('read sample indices file: ' + args.sample_indices_file)
    indices = read_hdf5_dataset(args.sample_indices_file)
    y_true = y_true[indices]
    y_pred = y_pred[indices]

    logger.info('save metrics file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with open(args.output_file, 'w') as f:
        f.write('r2\tmse\tpcc\n')
        f.write('%f'%r2_score(y_true, y_pred))
        f.write('\t%f'%mean_squared_error(y_true, y_pred))
        f.write('\t%f'%pearsonr(y_true, y_pred)[0])
        f.write('\n')

def generate_cv_masks(sample_indices, parent_table, k_female=10, k_male=10):
    """
    Generate cross-validation masks such that the parents of the test samples are different from training samples
    :param sample_indices: indices of samples to use
    :param parent_table: a 2D array of sample indices with shape (n_males, n_females)
    :param k_female: number of folds for female
    :param k_male: number of folds for male
    :return: a tuple: train_masks, test_masks
        Each element of the tuple is a 2D boolean array of shape (n_cv, n_females*n_males).
        The elements of the mask arrays indicates whether to use the sample.
    """
    from sklearn.model_selection import KFold
    n_samples_total = np.prod(parent_table.shape)
    all_indices = np.full(n_samples_total, np.nan)
    all_indices[sample_indices] = sample_indices
    indices_table = np.take(all_indices, parent_table)
    ind_female = np.nonzero(~np.all(np.isnan(indices_table), axis=0))[0]
    ind_male = np.nonzero(~np.all(np.isnan(indices_table), axis=1))[0]
    notnan = np.zeros(np.prod(parent_table.shape), dtype='bool')
    notnan[sample_indices] = True

    kfold_female = KFold(k_female, shuffle=True)
    kfold_male = KFold(k_male, shuffle=True)
    train_masks = []
    test_masks = []
    for train_index_female, _ in kfold_female.split(ind_female):
        train_index_female = ind_female[train_index_female]
        for train_index_male, _ in kfold_male.split(ind_male):
            train_index_male = ind_male[train_index_male]
            parent_table_train = np.ravel(parent_table[:, train_index_female][train_index_male])
            train_index = parent_table_train[notnan[parent_table_train]]
            train_mask = np.zeros(n_samples_total, dtype='bool')
            train_mask[train_index] = True
            train_masks.append(train_mask)

            parent_table_test = np.setdiff1d(np.ravel(sample_indices), parent_table_train)
            test_index = parent_table_test[notnan[parent_table_test]]
            test_mask = np.zeros(n_samples_total, dtype='bool')
            test_mask[test_index] = True
            test_masks.append(test_mask)

    train_masks = np.concatenate(train_masks).reshape((-1, n_samples_total))
    test_masks = np.concatenate(test_masks).reshape((-1, n_samples_total))
    return train_masks, test_masks

def linear_cv(X, y, sample_indices, parent_table, k_female=10, k_male=10):
    import numpy as np
    from sklearn.model_selection import KFold

    train_masks, test_masks = generate_cv_masks(sample_indices, parent_table,
                                                k_female=k_female, k_male=k_male)

    mix_factors = np.linspace(0.0, 1.0, 201, endpoint=True)
    # a 2D array of shape (n_mix_factors, n_samples)
    X_mixed = X[:, 0][np.newaxis, :]*(1.0 - mix_factors[:, np.newaxis]) + X[:, 1][np.newaxis, :]*mix_factors[:, np.newaxis]

    mse_train = np.zeros((len(mix_factors), train_masks.shape[0]))
    mse_test = np.zeros((len(mix_factors), train_masks.shape[0]))
    for i, mix_factor in enumerate(mix_factors):
        mse_train[i], mse_test[i] = fast_linregress_cv(X_mixed[i], y, train_masks, test_masks)
        #logger.info('mix_factor = %f, mse_train = %f, mse_test = %f'%(mix_factor, mse_train[i].mean(), mse_test[i].mean()))
    return mix_factors, mse_train, mse_test

def mixed_model(args):
    import h5py
    import numpy as np
    import dill as pickle
    from utils import read_hdf5_dataset, read_hdf5_single
    from sklearn.metrics import r2_score, mean_squared_error

    logger.info('read predictions of the first model: ' + args.input_file1)
    with h5py.File(args.input_file1, 'r') as f:
        X1 = f['y_pred'][:][:, np.newaxis]
    logger.info('read predictions of the second model: ' + args.input_file2)
    with h5py.File(args.input_file2, 'r') as f:
        X2 = f['y_pred'][:][:, np.newaxis]
    X = np.concatenate([X1, X2], axis=1)
    logger.info('read phenotypes from file: ' + args.phenotype_file)
    y = read_hdf5_dataset(args.phenotype_file)
    logger.info('read parent table from file: ' + args.parent_table_file)
    parent_table = read_hdf5_single(args.parent_table_file)

    if args.sample_indices_file:
        logger.info('read sample indices from: ' + args.sample_indices_file)
        sample_indices = read_hdf5_dataset(args.sample_indices_file)
    else:
        sample_indices = np.nonzero(~np.isnan(y))[0]
    X_train = X[sample_indices]
    y_train = y[sample_indices]

    logger.info('use model ' + args.model_name)
    logger.info('X.shape = %s, y.shape = %s' % (repr(X.shape), repr(y.shape)))
    if args.model_name == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X)
        y_pred_train = y_pred[sample_indices]
        logger.info('coefficients: ' + ', '.join([str(a) for a in model.coef_]))
    elif args.model_name == 'linear_cv':
        mix_factors, mse_train, mse_test = linear_cv(X, y, sample_indices, parent_table)
        best_mix_factor = mix_factors[np.argmin(mse_test.mean(axis=1))]
        logger.info('best mix factor: %f'%best_mix_factor)
        X_mixed = (X[:, 0]*(1 - best_mix_factor) + X[:, 1]*best_mix_factor)[:, np.newaxis]
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_mixed[sample_indices], y_train)
        y_pred = model.predict(X_mixed)
        y_pred_train = y_pred[sample_indices]
    else:
        raise ValueError('unknown model name: ' + args.model_name)

    logger.info('r2 score = %f' % r2_score(y_train, y_pred_train))
    logger.info('mse = %f' % mean_squared_error(y_train, y_pred_train))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_file = os.path.join(args.output_dir, 'model')
    logger.info('save model file: ' + model_file)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    pred_file = os.path.join(args.output_dir, 'predictions')
    logger.info('save predictions to file: ' + pred_file)
    with h5py.File(pred_file, 'w') as f:
        f.create_dataset('y_true', data=y)
        f.create_dataset('y_pred', data=y_pred)
        f.create_dataset('y_pred_train', data=y_pred_train)
        f.create_dataset('indices_train', data=sample_indices)
        if args.model_name == 'linear_cv':
            f.create_dataset('mse_train', data=mse_train)
            f.create_dataset('mse_test', data=mse_test)
            f.create_dataset('mix_factors', data=mix_factors)

def run_mixed_ridge(args):
    from utils import read_hdf5_dataset
    import h5py
    from models import MixedRidge
    import dill as pickle
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
        #U = U*S[np.newaxis, :]
        U = U[:, S**2 > 0.5]
        U = standardize_genotypes(U)
    logger.info('read phenotype file: ' + args.phenotype_file)
    y = read_hdf5_dataset(args.phenotype_file)
    logger.info('read parent table file: ' + args.parent_table_file)
    parent_table = read_hdf5_single(args.parent_table_file)
    logger.info('read training indices file: ' + args.train_index_file)
    train_index = read_hdf5_dataset(args.train_index_file)
    logger.info('read test indices file: ' + args.test_index_file)
    test_index = read_hdf5_dataset(args.test_index_file)

    indices_table, mask = get_indices_table(train_index, parent_table)
    if args.cv_type == 's1f':
        if args.k is None:
            k = parent_table.shape[0]
        else:
            k = args.k
        train_index_list, test_index_list, s0_index_list = cv_split_emaize(indices_table, mask,
                                                            k=k, method='s1f')
    elif args.cv_type == 's0':
        train_index_list, test_index_list, s0_index_list = cv_split_emaize(indices_table, mask,
                                                            k1=parent_table.shape[0]/5,
                                                            k2=parent_table.shape[1]/5,
                                                            method='s0')
    elif args.cv_type == 's1m':
        if args.k is None:
            k = parent_table.shape[1]
        else:
            k = args.k
        train_index_list, test_index_list, s0_index_list = cv_split_emaize(indices_table, mask,
                                                            k=k, method='s1m')
    else:
        raise ValueError('unkown cross-validation type: %s'%args.cv_type)
    logger.info('%d rows and %d columns in the indices table'%(indices_table.shape[0], indices_table.shape[1]))
    logger.info('number of cross-validation folds %d' % len(train_index_list))

    logger.info('number principal components to use: %d'%U.shape[1])
    alpha_list = [0.001, 0.01, 0.1]
    gamma_list = [0.0, 0.05, 0.1, 0.15, 0.2]
    alpha_list = [0.001]
    gamma_list = [0.15]
    alpha_list = [float(a) for a in args.alphas.split(',')]
    gamma_list = [float(a) for a in args.gammas.split(',')]
    metrics = {'pcc_cv': np.zeros((len(alpha_list), len(gamma_list), len(test_index_list))),
               'mse_cv': np.zeros((len(alpha_list), len(gamma_list), len(test_index_list)))}
    X_train, U_train, y_train = X[train_index], U[train_index], y[train_index]
    n_samples_total = np.prod(parent_table.shape)

    test_index_mask = np.zeros((len(test_index_list), n_samples_total), dtype='bool')
    if args.cv_type == 's0':
        for i in range(len(s0_index_list)):
            test_index_mask[i, train_index[s0_index_list[i]]] = True
    else:
        for i in range(len(test_index_list)):
            test_index_mask[i, train_index[test_index_list[i]]] = True
    for i, alpha in enumerate(alpha_list):
        #model.optimize_grid(X[train_index], U[train_index], y[train_index])
        '''
        mse_cv = np.zeros(len(train_index_list))
        pcc_cv = np.zeros(len(train_index_list))
        for j in range(len(train_index_list)):
            model.fit(X_train[train_index_list[j]],
                      U_train[train_index_list[j]],
                      y_train[train_index_list[j]],
                      gamma=0.1)
            y_pred_cv = model.predict(X_train[test_index_list[j]],
                                      U_train[test_index_list[j]])
            mse_cv[j] = mean_squared_error(y_train[test_index_list[j]], y_pred_cv)
            pcc_cv[j] = pearsonr(y_train[test_index_list[j]], y_pred_cv)[0]
        logger.info('cross-validation (real) MSE = %f, %f, %f'%(np.nanmin(mse_cv), np.nanmean(mse_cv), np.nanmax(mse_cv)))
        logger.info('cross-validation (real) PCC = %f, %f, %f' % (np.nanmin(pcc_cv), np.nanmean(pcc_cv), np.nanmax(pcc_cv)))
        '''
        for j, gamma in enumerate(gamma_list):
            model = MixedRidge(alphas=alpha)
            model.fit(X_train, U_train, y_train, gamma=gamma, cv=True)
            mse_cv = model.kfold(test_index_list, subset_indices=s0_index_list, return_mean=False)
            pcc_cv = model.pcc_cv
            metrics['pcc_cv'][i, j] = pcc_cv
            metrics['mse_cv'][i, j] = mse_cv
            logger.info('cross-validation (fast) MSE = %f, %f, %f' % (np.nanmin(mse_cv), np.nanmean(mse_cv), np.nanmax(mse_cv)))
            logger.info('cross-validation (fast) PCC = %f, %f, %f' % (np.nanmin(pcc_cv), np.nanmean(pcc_cv), np.nanmax(pcc_cv)))
            logger.info('alpha=%f, gamma=%f' % (alpha, gamma))

    pcc_cv_mean = np.nanmean(metrics['pcc_cv'], axis=2)
    i_best = np.argmax(np.max(pcc_cv_mean, axis=1))
    best_alpha = alpha_list[i_best]
    best_gamma = gamma_list[np.argmax(pcc_cv_mean[i_best])]
    logger.info('best model: alpha=%f, gamma=%f' % (best_alpha, best_gamma))

    best_model = MixedRidge(alphas=best_alpha)
    best_model.fit(X_train, U_train, y_train, gamma=best_gamma)
    y_pred_best = best_model.predict(X, U)
    '''
    logger.info('best model on test data: pcc=%f, mse=%f'%(
        pearsonr(y[test_index], y_pred_best[test_index])[0],
        mean_squared_error(y[test_index], y_pred_best[test_index])))
        '''
    logger.info('save results to output directory: ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #model_file = os.path.join(args.output_dir, 'model')
    #with open(model_file, 'wb') as f:
    #    pickle.dump(best_model, f)
    pred_file = os.path.join(args.output_dir, 'predictions')
    with h5py.File(pred_file, 'w') as f:
        f.create_dataset('best_alpha', data=best_alpha)
        f.create_dataset('best_gamma', data=best_gamma)
        f.create_dataset('y_true', data=y)
        f.create_dataset('y_pred', data=y_pred_best)
        f.create_dataset('pcc_cv', data=metrics['pcc_cv'])
        f.create_dataset('mse_cv', data=metrics['mse_cv'])
        f.create_dataset('alpha_list', data=np.asarray(alpha_list))
        f.create_dataset('gamma_list', data=np.asarray(gamma_list))
        f.create_dataset('test_index_mask', data=test_index_mask)

def select_best_subset(args):
    import h5py
    from tqdm import tqdm
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    from utils import prepare_output_file, read_hdf5_dataset

    logger.info('read sample indices of test dataset from ' + args.test_index_file)
    test_index = read_hdf5_dataset(args.test_index_file)

    traits = args.traits.split(',')
    def iterator():
        cv_type = 's1f'
        for gamma in args.gammas.split(','):
            for n_snps in [int(a) for a in args.n_snps.split(',')]:
                for trait in traits:
                    for snp_set in range(args.n_groups):
                        filename = args.input_dir + '/gamma={gamma}/{n_snps}/{trait}/{snp_set}/{cv_type}/predictions'.format(
                            n_snps=n_snps, trait=trait, snp_set=snp_set, cv_type=cv_type, gamma=gamma)
                        yield ('random_choice', gamma, trait, n_snps, snp_set, cv_type, filename)

    def query_dict(records, **kwargs):
        '''
        Search for records (dicts) that match the key-value pairs
        :param records: a list of dicts
        :param kwargs: key-value pairs
        :return: a list of records that match the query arguments
        '''
        results = []
        for record in records:
            val = True
            for key in kwargs:
                val = val and (record[key] == kwargs[key])
            if val:
                results.append(record)
        return results

    logger.info('read prediction results')
    predictions = []
    for method, gamma, trait, n_snps, snp_set, cv_type, filename in tqdm(list(iterator())):
        with h5py.File(filename, 'r') as f:
            predictions.append({'trait': trait,
                                'gamma': f['best_gamma'][()],
                                'alpha': f['best_alpha'][()],
                                'method': method,
                                'n_snps': n_snps,
                                'snp_set': snp_set,
                                'y_pred': f['y_pred'][:],
                                'cv_type': cv_type,
                                'mse_cv': np.ravel(f['mse_cv']),
                                'pcc_cv': np.ravel(f['pcc_cv'])})

    logger.info('summarize cross-validation metrics')
    summary = []
    for pred in tqdm(predictions):
        summary.append((pred['method'], pred['gamma'], pred['alpha'],
                        pred['trait'], pred['n_snps'], pred['snp_set'], pred['cv_type'],
                        np.min(pred['pcc_cv']), np.min(pred['mse_cv']),
                        np.mean(pred['pcc_cv']), np.mean(pred['mse_cv']),
                        np.max(pred['pcc_cv']), np.max(pred['mse_cv']),
                        np.median(pred['pcc_cv']), np.median(pred['mse_cv'])))

    summary = pd.DataFrame.from_records(summary,
                                        columns=('method', 'gamma', 'alpha',
                                                 'trait', 'n_snps', 'snp_set', 'cv_type',
                                                 'pcc_cv_min', 'mse_cv_min',
                                                 'pcc_cv_mean', 'mse_cv_mean',
                                                 'pcc_cv_max', 'mse_cv_max',
                                                 'pcc_cv_median', 'mse_cv_median'))
    ascending = args.by.startswith('mse')
    summary_best = summary.sort_values(['trait', args.by], ascending=ascending).groupby(
        ['trait']).head(3)

    if not os.path.exists(args.output_dir):
        logger.info('create output directory: ' + args.output_dir)
        os.makedirs(args.output_dir)

    summary_file = os.path.join(args.output_dir, 'summary.txt')
    logger.info('write summary of all SNP subsets to file: ' + summary_file)
    summary.to_csv(summary_file, sep='\t', index=False)

    summary_best_file = os.path.join(args.output_dir, 'summary_best.txt')
    logger.info('write summary of best SNP subsets to file: ' + summary_best_file)
    summary_best.to_csv(summary_best_file, sep='\t', index=False)

    logger.info('extract predictions from best SNP subsets')
    rank = {}
    for trait in traits:
        rank[trait] = 0
    for index, record in summary_best.iterrows():
        pred_file = args.input_dir + '/gamma={gamma:.2f}/{n_snps}/{trait}/{snp_set}/{cv_type}/predictions'.format(**record.to_dict())
        with h5py.File(pred_file, 'r') as f:
            y_pred = f['y_pred'][:]
        trait = record['trait']
        test_pred_file = os.path.join(args.output_dir, 'prediction.%s.%d.txt'%(trait, rank[trait]))
        logger.info('save test predictions to file: ' + test_pred_file)
        np.savetxt(test_pred_file, y_pred[test_index])
        rank[trait] += 1

def plot_predictions(args):
    import h5py
    from utils import read_hdf5_single, prepare_output_file, read_hdf5_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    import numpy as np

    def normalize_phenotype(x, range_pheno=4.0):
        return (np.clip(x, -range_pheno, range_pheno) + range_pheno) / 2.0 / range_pheno

    logger.info('read parent table file: ' + args.parent_table_file)
    parent_table = read_hdf5_single(args.parent_table_file)
    logger.info('read predictions from file: ' + args.input_file)
    with h5py.File(args.input_file, 'r') as f:
        y_true = f['y_true'][:]
        y_pred = f['y_pred'][:]
    logger.info('read training indices from file: ' + args.train_indices_file)
    train_index = read_hdf5_dataset(args.train_indices_file)
    logger.info('read test indices from file: ' + args.test_indices_file)
    test_index = read_hdf5_dataset(args.test_indices_file)

    y_pred_train = np.full(y_pred.shape, np.nan)
    y_pred_train[train_index] = y_pred[train_index]
    y_pred_test = np.full(y_pred.shape, np.nan)
    y_pred_test[test_index] = y_pred[test_index]

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with PdfPages(args.output_file) as pdf:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
        axes[0].matshow(np.take(np.ravel(normalize_phenotype(y_true)), parent_table), cmap=plt.cm.RdBu_r)
        axes[0].set_title('True phenotypes')

        axes[1].matshow(np.take(np.ravel(normalize_phenotype(y_pred)), parent_table), cmap=plt.cm.RdBu_r)
        axes[1].set_title('Predicted phenotypes')

        axes[2].matshow(np.take(np.ravel(normalize_phenotype(y_pred_train)), parent_table), cmap=plt.cm.RdBu_r)
        axes[2].set_title('Predicted phenotypes (train)')

        axes[3].matshow(np.take(np.ravel(normalize_phenotype(y_pred_test)), parent_table), cmap=plt.cm.RdBu_r)
        axes[3].set_title('Predicted phenotypes (test)')

        plt.tight_layout()
        pdf.savefig(fig)

        plt.clf()
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes[0, 0].hist(y_true[~np.isnan(y_true)], bins=50)
        axes[0, 0].set_title('True phenotypes')
        axes[0, 1].hist(y_true[train_index], bins=50)
        axes[0, 1].set_title('True phenotypes (train)')
        axes[0, 2].hist(y_true[test_index], bins=50)
        axes[0, 2].set_title('True phenotypes (test)')
        axes[1, 0].hist(y_pred, bins=50)
        axes[1, 0].set_title('Predicted phenotypes')
        axes[1, 1].hist(y_pred[train_index], bins=50)
        axes[1, 1].set_title('Predicted phenotypes (train)')
        axes[1, 2].hist(y_pred[test_index], bins=50)
        axes[1, 2].set_title('Predicted phenotypes (test)')
        for i in range(2):
            for j in range(3):
                axes[i, j].set_xlim(-5, 5)
        plt.tight_layout()
        pdf.savefig(fig)

        plt.clf()
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].scatter(y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_true)], s=3)
        axes[0].set_xlabel('True phenotypes')
        axes[0].set_ylabel('Predicted phenotypes')
        axes[0].set_title('All samples')

        axes[1].scatter(y_true[train_index], y_pred[train_index], s=3)
        axes[1].set_xlabel('True phenotypes')
        axes[1].set_ylabel('Predicted phenotypes')
        axes[1].set_title('Training samples')

        axes[2].scatter(y_true[test_index], y_pred[test_index], s=3)
        axes[2].set_xlabel('True phenotypes')
        axes[2].set_ylabel('Predicted phenotypes')
        axes[2].set_title('Training samples')

        plt.tight_layout()
        pdf.savefig(fig)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Commands for mixed models')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('single_model')
    parser.add_argument('--genotype-file', '-i', type=str, required=True,
                        help='HDF5 filename and dataset name separated by ":". Assume rows are samples and columns are features.')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='phenotype file. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--transpose-x', action='store_true',
                        help='transpose the rows and columns of X first')
    parser.add_argument('--normalize-x', action='store_true',
                        help='normalize X to zero means and unit variances')
    parser.add_argument('--sample-indices-file', type=str,
                        help='indices of samples to use. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--parent-table-file', type=str, required=True,
                        help='HDF5 filename')
    parser.add_argument('--feature-indices-file', type=str,
                        help='indices of features to use. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--model-name', type=str, default='ridge', choices=('gpr', 'ridge', 'ridge_cv'),
                        help='name the model to use')
    parser.add_argument('--output-residuals', action='store_true',
                        help='save residuals to the prediction file')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='prefix for output filenames (output files: model, predictions)')

    parser = subparsers.add_parser('mixed_model')
    parser.add_argument('--input-file1', '-a', type=str, required=True,
                        help='predictions of the first model')
    parser.add_argument('--input-file2', '-b', type=str, required=True,
                        help='predictions of the second model')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='phenotype file. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--parent-table-file', type=str, required=True,
                        help='an HDF5 file with females as columns and males as rows')
    parser.add_argument('--sample-indices-file', type=str,
                        help='indices of samples to use. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--model-name', type=str, default='linear', choices=('linear', 'linear_cv'),
                        help='name the model to use')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='prefix for output filenames (output files: model, predictions)')

    parser = subparsers.add_parser('evaluate')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='prediction file produced by "train"')
    parser.add_argument('--sample-indices-file', type=str, required=True,
                        help='indices of samples to evaluate')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file')

    parser = subparsers.add_parser('plot_predictions')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='prediction file produced by "train"')
    parser.add_argument('--parent-table-file', type=str, required=True,
                       help='HDF5 filename')
    parser.add_argument('--train-indices-file', type=str, required=True,
                        help='HDF5 filename')
    parser.add_argument('--test-indices-file', type=str, required=True,
                        help='HDF5 filename')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file')

    parser = subparsers.add_parser('anova_linregress')
    parser.add_argument('--genotype-file', '-i', type=str, required=True,
                        help='HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='phenotype file. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--indices-file', type=str,
                        help='indices of samples to use. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--output-file', '-o', type=str, required=True)

    parser = subparsers.add_parser('mixed_ridge')
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
    parser.add_argument('--gammas', type=str, default='0.05',
                        help='comma-separated list of gamma values to search')
    parser.add_argument('--alphas', type=str, default='0.001',
                        help='comma-separated list of gamma values to search')
    parser.add_argument('-k', type=int, help='number of CV folds')
    parser.add_argument('--parent-table-file', type=str, required=True,
                        help='parent table in HDF5 format')
    parser.add_argument('--train-index-file', type=str, required=True,
                        help='training sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--test-index-file', type=str, required=True,
                        help='test sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='output directory')

    parser = subparsers.add_parser('select_best_subset')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='output directory of mixed_ridge')
    parser.add_argument('--test-index-file', type=str, required=True,
                        help='test sample indices. HDF5 filename and dataset name separated by ":"')
    parser.add_argument('--genotype-dir', type=str, required=False,
                        help='directory that contains the features. File names are n_snps. e.g. 100, 200.')
    parser.add_argument('--n-snps', type=str, default='')
    parser.add_argument('--n-groups', type=int, default=1)
    parser.add_argument('--gammas', type=str, default='')
    parser.add_argument('--traits', type=str, default='trait1,trait2,trait3',
                        help='comma-separated list of traits')
    parser.add_argument('--by', type=str, default='mse_cv_mean',
                        choices=('pcc_cv_min', 'mse_cv_min', 'pcc_cv_mean', 'mse_cv_mean',
                                 'pcc_cv_max', 'mse_cv_max', 'pcc_cv_median', 'mse_cv_median'),
                        help='select best subset according to the metric')
    parser.add_argument('--output-dir', '-o', type=str, required=True)

    args = main_parser.parse_args()

    import numpy as np

    logger = logging.getLogger('run_mixed_model.' + args.command)
    if args.command == 'single_model':
        single_model(args)
    elif args.command == 'mixed_model':
        mixed_model(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'plot_predictions':
        plot_predictions(args)
    elif args.command == 'mixed_ridge':
        run_mixed_ridge(args)
    elif args.command == 'select_best_subset':
        select_best_subset(args)
