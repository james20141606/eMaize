#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')
logger = logging.getLogger('random_projection')

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Command line interface for sklearn.random_projection')
    subparsers = main_parser.add_subparsers(dest='command')
    # command: generate
    parser = subparsers.add_parser('generate', help='generate a random sparse matrix')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--input-file', type=str,
                       help='a HDF5 file')
    group.add_argument('-p', '--n-features', type=int,
                       help='number of features')
    parser.add_argument('--dataset', type=str,
                                 help='dataset name in the HDF5 file')
    parser.add_argument('--transpose', action='store_true',
                                 help='transpose the matrix in the HDF5 file before random projection')
    parser.add_argument('-r', '--n-components', type=int, required=True,
                                 help='number of components after random projection')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                                 help='matrix in .npz format')
    # command: test_load
    parser = subparsers.add_parser('test_load', help='test if a npz file can be loaded')
    parser.add_argument('input_file', type=str,
                             help='input matrix file in .npz format')

    # command: transform
    parser = subparsers.add_parser('transform',
                                             help='transform an input matrix using random projection')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                                  help='data matrix')
    parser.add_argument('--datasets', type=str, required=True,
                                  help='comma-separated list of dataset names. * for all datasets.')
    parser.add_argument('--components-file', type=str,
                                  help='components in .npz format')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                                  help='transformed features file')
    parser.add_argument('--merge', action='store_true',
                                  help='merge transformed features into a single matrix')
    parser.add_argument('--output-dataset', type=str, default='data',
                                  help='output dataset name')
    # command: normalize
    parser = subparsers.add_parser('normalize',
                                   help='normalize the transformed features into z-scores')
    parser.add_argument('-i', '--input-file', type=str, required=True,
                         help='output file of the transform command')
    parser.add_argument('--scaler-file', type=str, required=True,
                        help='input file containing scales for each feature (HDF5 file with dataset mean_ and scale_)')
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help='normalized features file')
    # command: merge
    args = main_parser.parse_args()

    import numpy as np
    from sklearn.random_projection import SparseRandomProjection
    from scipy.sparse import load_npz, save_npz
    if args.command == 'generate':
        if args.input_file is not None:
            import h5py
            f = h5py.File(args.input_file, 'r')
            if args.dataset is None:
                raise ValueError('option --dataset is required for HDF5 input file')
            if args.transpose:
                n_features = f[args.dataset].shape[0]
            else:
                n_features = f[args.dataset].shape[1]
        if args.n_features is not None:
            n_features = args.n_features
        logger.info('number of features: %d'%n_features)
        X = np.zeros((2, n_features))
        proj = SparseRandomProjection(args.n_components)
        logger.info('generate random projection matrix (%d components)'%args.n_components)
        proj.fit(X)
        logger.info('save random projection matrix to ' + args.output_file)
        prepare_output_file(args.output_file)
        save_npz(args.output_file, proj.components_, compressed=False)

    elif args.command == 'test_load':
        load_npz(args.input_file)

    elif args.command == 'transform':
        import h5py
        logger.info('load random projection components from ' + args.components_file)
        components = load_npz(args.components_file)
        proj = SparseRandomProjection(components.shape[0])
        proj.components_ = components
        fin = h5py.File(args.input_file)
        y = {}
        if args.datasets == '*':
            datasets = fin.keys()
        else:
            datasets = args.datasets.split(',')
        for dataset in datasets:
            X = fin[dataset][:].reshape((1, -1))
            logger.info('transform dataset ' + dataset)
            y[dataset] = np.ravel(proj.transform(X))
            del X
        fin.close()
        logger.info('save transformed features to ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file)
        if args.merge:
            logger.info('merge transformed features')
            n_samples = len(y)
            y = np.concatenate([y[dataset] for dataset in datasets]).reshape((n_samples, -1))
            fout.create_dataset('data', data=y)
        else:
            logger.info('save transformed features as separate datasets')
            for dataset in datasets:
                fout.create_dataset(dataset, data=y[dataset])
        fout.close()

    elif args.command == 'normalize':
        import h5py
        from sklearn.preprocessing import StandardScaler

        logger.info('read scaler file: ' + args.scaler_file)
        fin = h5py.File(args.scaler_file, 'r')
        scaler = StandardScaler(copy=False)
        scaler.mean_ = fin['mean_'][:]
        scaler.scale_ = fin['scale_'][:]
        fin.close()

        logger.info('read input file: ' + args.input_file)
        fin = h5py.File(args.input_file, 'r')
        logger.info('create output file: ' + args.output_file)
        prepare_output_file(args.output_file)
        fout = h5py.File(args.output_file, 'w')
        for dataset in fin.keys():
            logger.info('normalize dataset ' + dataset)
            data = fin[dataset][:].reshape((1, -1))
            data = scaler.transform(data)
            fout.create_dataset(dataset, data=np.ravel(data))
        fin.close()
        fout.close()



