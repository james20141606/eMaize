#! /usr/bin/env python
import argparse, sys, os, errno
import dill as pickle
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('hyperopt_search')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel optimization of hyper-parameters using hyperopt-sklearn and MongoDB')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='input file in HDF5 format')
    parser.add_argument('--index-file', type=str, required=False,
                        help='indices for selecting a subset of the input dataset (HDF5 format)')
    parser.add_argument('-x', '--xname', type=str, default='X', help='dataset name for X')
    parser.add_argument('-y', '--yname', type=str, default='y', help='dataset name for y')
    parser.add_argument('-k', '--n-folds', type=int,
                        help='number of folds for K-fold cross-validation')
    parser.add_argument('--mongo', type=str,
                        help='connection string for MongoDB (e.g. mongo://<host>:<port>/<db>')
    parser.add_argument('--name', type=str, default='job', help='job identifier')
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument('--classifier', type=str, help='classifier name')
    g.add_argument('--regressor', type=str, help='regressor name')
    parser.add_argument('--algo', type=str, default='rand.suggest',
                        choices=('rand.suggest', 'tpe.suggest'),
                        help='hyperopt search algorithm')
    parser.add_argument('--max-evals', type=int, default=10,
                        help='maximum number of hyper-parameters to try')
    parser.add_argument('-o', '--output-dir', type=str, help='output directory')
    args = parser.parse_args()

    import h5py
    import numpy as np
    import hyperopt.tpe
    from hyperopt.mongoexp import MongoTrials
    import hpsklearn
    import joblib

    logger.info('read input file: ' + args.input_file)
    logger.info('X name: %s, y name: %s'%(args.xname, args.yname))
    fin = h5py.File(args.input_file, 'r')
    X = fin[args.xname][:]
    y = fin[args.yname][:]
    fin.close()

    if args.index_file is not None:
        logger.info('read index file: ' + args.index_file)
        f = h5py.File(args.index_file, 'r')
        indices = f[f.keys()[0]][:]
        f.close()
        X = np.take(X, indices, axis=0)
        y = np.take(y, indices, axis=0)
        del indices

    logger.info('create hyperopt estimator')
    classifier = None
    regressor = None
    if args.classifier is not None:
        if args.classifier == 'any':
            classifier = hpsklearn.components.any_classifier(args.name + '.classifier')
        else:
            classifier = getattr(hpsklearn.components, args.classifier)(args.name + '.classifier')
    else:
        if args.regressor == 'any':
            regressor = hpsklearn.components.any_regressor(args.name + '.regressor')
        else:
            regressor = getattr(hpsklearn.components, args.regressor)(args.name + '.regressor')
    algos = {'rand.suggest': hyperopt.rand.suggest,
             'tpe.suggest': hyperopt.tpe.suggest}
    algo = algos[args.algo]
    estimator = hpsklearn.HyperoptEstimator(classifier=classifier,
                                            regressor=regressor,
                                            algo=algo,
                                            max_evals=args.max_evals)

    logger.info('fit hyperopt estimator')
    trials = None
    warm_start = False
    if args.mongo is not None:
        logger.info('connect to MongoDB at ' + args.mongo)
        trials = MongoTrials(args.mongo)
        warm_start = True
        estimator.trials = trials
    estimator.fit(X, y, n_folds=args.n_folds, warm_start=warm_start)

    best_model_file = os.path.join(args.output_dir, 'best_model')
    logger.info('save the best model to ' + best_model_file)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(best_model_file, 'w') as f:
        pickle.dump(estimator.best_model_, f)

