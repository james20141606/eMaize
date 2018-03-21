#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')

def phenotypes_to_hdf5(args):
    import pandas as pd
    import h5py
    from utils import prepare_output_file

    logger.info('read phenotype file: ' + args.input_file)
    phenotypes = pd.read_table(args.input_file)
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        for col in phenotypes.columns:
            if phenotypes[col].dtype == 'O':
                f.create_dataset(col, data=phenotypes[col].values.astype('S'))
            else:
                f.create_dataset(col, data=phenotypes[col].values)

def extract_snp_pos(args):
    import h5py
    import subprocess
    import numpy as np
    from utils import prepare_output_file

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    for i in range(1, 11):
        genotype_file = os.path.join(args.input_dir, 'chr%d_emaize.genoMat'%i)
        logger.info('read genotype matrix file: ' + genotype_file)
        p = subprocess.Popen(['awk', 'NR>1{print $4}', genotype_file], stdout=subprocess.PIPE)
        positions = np.loadtxt(p.stdout, dtype='int64')
        fout.create_dataset('chr%d'%i, data=positions)
    fout.close()

def create_gsm(args):
    import h5py
    import numpy as np
    from utils import prepare_output_file, read_hdf5_dataset

    '''
    logger.info('read genomic positions from file: ' + args.genomic_pos_file)
    positions = {}
    with h5py.File(args.genomic_pos_file, 'r') as f:
        for i in range(1, 11):
            positions['chr%d'%i] = f['chr%d'%i][:]
    n_snps_per_chrom = {chrom:positions[chrom].shape[0] for chrom in positions.keys()}
    n_snps_total = sum(n_snps_per_chrom.values())
    X = []
    for chrom in positions.keys():
        genotype_file = os.path.join(args.input_dir, chrom)
        logger.info('read genotype file: ' + genotype_file)
        with h5py.File(genotype_file, 'r') as f:
            n_sel = int(np.round(args.n_snps*float(n_snps_per_chrom[chrom])/n_snps_total))
            ind = np.random.choice(n_snps_per_chrom[chrom], size=n_sel)
            X.append(f['data'][:][ind])
    X = np.concatenate(X, axis=0).astype('float32')
    '''
    logger.info('read genotypes from file: ' + args.input_file)
    X = read_hdf5_dataset(args.input_file).astype('float64')
    logger.info('number of selected SNPs: %d'%X.shape[0])
    logger.info('calculate GSM')
    X -= X.mean(axis=1)[:, np.newaxis]
    X_std = np.sqrt(np.sum(X**2, axis=1))
    X_std[np.isclose(X_std, 0.0)] = 1.0
    X = X/X_std[:, np.newaxis]
    logger.info('calculate K')
    K = np.dot(X.T, X)
    logger.info('run SVD on X')
    U, S, V = np.linalg.svd(X.T, full_matrices=False)
    V = V.T
    logger.info('save GSM to file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('K', data=K)
        f.create_dataset('U', data=U)
        f.create_dataset('S', data=S)
        f.create_dataset('V', data=V)

def random_select(args):
    import h5py
    import numpy as np
    from utils import prepare_output_file

    logger.info('read genomic positions from file: ' + args.genomic_pos_file)
    positions = {}
    with h5py.File(args.genomic_pos_file, 'r') as f:
        for i in range(1, 11):
            positions['chr%d' % i] = f['chr%d' % i][:]
    n_snps_per_chrom = {chrom: positions[chrom].shape[0] for chrom in positions.keys()}
    n_snps_total = sum(n_snps_per_chrom.values())

    X = [[] for i in range(args.n_select)]
    chroms = [[] for i in range(args.n_select)]
    positions_sel = [[] for i in range(args.n_select)]
    n_sel_total = 0
    for i_chrom in range(1, 11):
        chrom = 'chr%d' % i_chrom
        genotype_file = os.path.join(args.input_dir, chrom)
        logger.info('read genotype file: ' + genotype_file)
        with h5py.File(genotype_file, 'r') as f:
            n_sel = int(np.round(args.n_snps * float(n_snps_per_chrom[chrom]) / n_snps_total))
            logger.info('select %d SNPs on chromosome %d'%(n_sel, i_chrom))
            X_chrom = f['data'][:]
            for i_select in range(args.n_select):
                ind = np.random.choice(n_snps_per_chrom[chrom], size=n_sel, replace=False)
                X[i_select].append(X_chrom[ind])
                chroms[i_select].append(np.full(n_sel, i_chrom, dtype='int8'))
                positions_sel[i_select].append(positions[chrom][ind])
            del X_chrom
            n_sel_total += n_sel
    logger.info('number of SNPs selected: %d' % n_sel_total)

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    fout = h5py.File(args.output_file, 'w')
    for i_select in range(args.n_select):
        X[i_select] = np.concatenate(X[i_select], axis=0)
        chroms[i_select] = np.concatenate(chroms[i_select])
        positions_sel[i_select] = np.concatenate(positions_sel[i_select])
        g = fout.create_group(str(i_select))
        g.create_dataset('X', data=X[i_select])
        g.create_dataset('chrom', data=chroms[i_select])
        g.create_dataset('position', data=positions_sel[i_select])
    fout.close()

def run_generate_parent_table(args):
    from utils import generate_parent_table, prepare_output_file
    import numpy as np
    import h5py

    logger.info('read phenotypes from file: ' + args.input_file)
    parent_table = generate_parent_table(args.input_file)
    logger.info('save parent table to file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('data', data=parent_table)
    #np.savetxt(args.output_file, parent_table, fmt='%d')

def random_cv_split(args):
    import numpy as np
    import h5py
    from utils import read_hdf5_single, cv_split_emaize, get_indices_table, prepare_output_file, read_hdf5_dataset

    logger.info('read training indices file: ' + args.train_index_file)
    train_indices_all = read_hdf5_dataset(args.train_index_file)
    logger.info('read parent table file: ' + args.parent_table_file)
    parent_table = read_hdf5_single(args.parent_table_file)
    indices_table, mask = get_indices_table(train_indices_all, parent_table)

    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        for k in range(args.n_datasets):
            row_indices = np.random.choice(indices_table.shape[0], 5, replace=False)
            col_indices = np.random.choice(indices_table.shape[1], 5, replace=False)
            test_indices = np.union1d(indices_table[row_indices, :].reshape((-1,)),
                 indices_table[:, col_indices].reshape((-1,)))
            train_indices = np.setdiff1d(train_indices_all, test_indices)
            test_indices = np.intersect1d(test_indices, train_indices_all)
            train_indices = np.intersect1d(train_indices, train_indices_all)
            g = f.create_group(str(k))
            g.create_dataset('train', data=train_indices)
            g.create_dataset('test', data=test_indices)
    
def random_select_subset(args):
    import h5py
    import numpy as np
    from utils import prepare_output_file

    if ':' not in args.input_file:
        raise ValueError('missing group name in input file: ' + args.input_file)
    logger.info('read input file: ' + args.input_file)
    input_file, group_name = args.input_file.split(':')
    with h5py.File(input_file, 'r') as f:
        X = f['/%s/X'%group_name][:]
        chrom = f['/%s/chrom'%group_name][:]
        positions = f['/%s/position'%group_name][:]

    logger.info('select %d SNP subsets of size %d'%(args.n_groups, args.n_snps))
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        if args.method == 'random_choice':
            for i in range(args.n_groups):
                ind = np.random.choice(X.shape[0], size=args.n_snps, replace=False)
                f.create_dataset('/%d/X' % i, data=X[ind])
                f.create_dataset('/%d/chrom' % i, data=chrom[ind])
                f.create_dataset('/%d/position' % i, data=positions[ind])
        elif args.method == 'seq':
            for i in range(X.shape[0] / args.n_snps):
                ind = np.r_[(i * args.n_snps):((i + 1) * args.n_snps)]
                f.create_dataset('/%d/X' % i, data=X[ind])
                f.create_dataset('/%d/chrom' % i, data=chrom[ind])
                f.create_dataset('/%d/position' % i, data=positions[ind])

def phenotypes_to_train_test_indices(args):
    import pandas as pd
    import numpy as np
    import h5py
    from utils import prepare_output_file

    logger.info('read input file: ' + args.input_file)
    phenotypes = pd.read_table(args.input_file)
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('train', data=np.nonzero((phenotypes['type'] == 'training').values)[0])
        f.create_dataset('test', data=np.nonzero((phenotypes['type'] == 'test').values)[0])

def convert_train_test_indices(args):
    import numpy as np
    import h5py
    from utils import prepare_output_file

    logger.info('read training sample indices from file: ' + args.train_index_file)
    train_index = np.loadtxt(args.train_index_file, dtype='int')
    logger.info('read training sample indices from file: ' + args.test_index_file)
    test_index = np.loadtxt(args.test_index_file, dtype='int')
    logger.info('create output file: ' + args.output_file)
    prepare_output_file(args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('train', data=train_index)
        f.create_dataset('test', data=test_index)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Preprocessing script')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('phenotypes_to_hdf5')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='phenotype table file (emaize_data/phenotype/pheno_emaize.txt)')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='HDF5 file containing columns as datasets')

    parser = subparsers.add_parser('phenotypes_to_train_test_indices')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='phenotype table file (emaize_data/phenotype/pheno_emaize.txt)')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='HDF5 file containing two datasets: train, test')

    parser = subparsers.add_parser('generate_parent_table')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='phenotype table file (emaize_data/phenotype/pheno_emaize.txt)')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='a table of sample indices with females as columns and males as rows')

    parser = subparsers.add_parser('extract_snp_pos',
                                   help='convert 2-bit representation of genotype to minor allele copy numbers (0, 1, 2)')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='input directory containing raw genotype matrices (e.g. chr1_emaize.genoMat)')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format with chromosome names as dataset names')

    parser = subparsers.add_parser('random_select',
                                   help='randomly select SNPs in all chromosomes')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='input directory containing raw genotype matrices (e.g. chr1_emaize.genoMat)')
    parser.add_argument('--genomic-pos-file', type=str, required=True,
                        help='an HDF5 file containing genomic positions of each SNP')
    parser.add_argument('--n-snps', '-n', type=int, default=10000,
                        help='number of SNPs to sample uniformly from the whole genome')
    parser.add_argument('--n-select', '-k', type=int, default=1,
                        help='number of SNP sets to selectly randomly')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format with dataset names in format: <i_select>/X, chrom, position')

    parser = subparsers.add_parser('random_select_subset')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='output file of the random_select command. Format: filename:group_name')
    parser.add_argument('--n-snps', '-n', type=int, default=100,
                        help='number of SNPs to sample uniformly')
    parser.add_argument('--n-groups', '-k', type=int, default=200,
                        help='number of groups to select')
    parser.add_argument('--method', '-m', type=str, default='random_choice',
                        choices=('random_choice', 'seq'),
                        help='seq: select sequentially. random_choice: randomly select')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format with dataset names in format: <i_select>/X, chrom, position')

    parser = subparsers.add_parser('create_gsm',
                                   help='create a genetic similarity matrix from genotype data')
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='an HDF5 file containing genotypes (minor allele copy numbers) [n_snps, n_samples]')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format containing the similarity matrix (dataset=data)')

    parser = subparsers.add_parser('convert_train_test_indices')
    parser.add_argument('--train-index-file', type=str, required=True,
                        help='a plain text file containing training sample indices, one per line')
    parser.add_argument('--test-index-file', type=str, required=True,
                        help='a plain text file containing test sample indices, one per line')
    parser.add_argument('--output-file', '-o', type=str, required=True,
                        help='output file in HDF5 format with two datasets: train, test')
    
    parser = subparsers.add_parser('random_cv_split',
        help='randomly split samples preserving pedigree structure')
    parser.add_argument('--n-datasets', '-n', type=int, default=100)
    parser.add_argument('--train-index-file', type=str, required=True)
    parser.add_argument('--parent-table-file', type=str, required=True)
    parser.add_argument('--output-file', '-o', type=str, required=True)

    args = main_parser.parse_args()

    logger = logging.getLogger('preprocess.' + args.command)
    if args.command == 'extract_snp_pos':
        extract_snp_pos(args)
    elif args.command == 'create_gsm':
        create_gsm(args)
    elif args.command == 'random_select':
        random_select(args)
    elif args.command == 'phenotypes_to_hdf5':
        phenotypes_to_hdf5(args)
    elif args.command == 'generate_parent_table':
        run_generate_parent_table(args)
    elif args.command == 'random_select_subset':
        random_select_subset(args)
    elif args.command == 'phenotypes_to_train_test_indices':
        phenotypes_to_train_test_indices(args)
    elif args.command == 'convert_train_test_indices':
        convert_train_test_indices(args)
    elif args.command == 'random_cv_split':
        random_cv_split(args)