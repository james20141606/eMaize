#! /usr/bin/env python
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s [%(levelname)s] : %(message)s')

def get_snpdata(iid, genotype_file, sample_indices=None,
                transpose=False,
                snp_indices=None,
                std_filter_indices=None,
                noise=1e-6, shuffle=True):
    """

    :param iid: 2D array of string type: (n_snps,)
    :param genotype_file: an HDF5 file with three datasets: X, position and chrom
        X: copy numbers of minor alleles (n_snps, n_samples)
        position: genomic position on chromosomes (n_snps,)
        chrom: chromosomes IDs. 1D array of integer type (n_snps,).
    :param sample_indices: 1D array of integers for selecting samples
    :param transpose:
    :param snp_indices:
    :param std_filter_indices:
    :param noise:
    :param shuffle:
    :return:
    """
    from pysnptools.snpreader import SnpData
    group = None
    if ':' in genotype_file:
        i = genotype_file.index(':')
        group = genotype_file[(i + 1):]
        genotype_file = genotype_file[:i]

    with h5py.File(genotype_file, 'r') as f:
        if group is None:
            g = f
        else:
            g = f[group]
        X = g['X'][:]
        if transpose:
            X = X.T
        if sample_indices is not None:
            X = np.take(X, sample_indices, axis=1)
            iid = np.take(iid, sample_indices, axis=0)
        if shuffle:
            ind = np.random.permutation(X.shape[0])
            X = np.take(X, ind, axis=0)

        position = g['position'][:]
        chrom = g['chrom'][:]

        # only select a given subset of SNPs
        if snp_indices is not None:
            X = np.take(X, snp_indices, axis=0)
            position = np.take(position, snp_indices)
            chrom = np.take(chrom, snp_indices)

        # remove SNPs with standard deviation smaller than a threshold
        if std_filter_indices is not None:
            index_not_constant = np.nonzero(X[:, std_filter_indices].std(axis=1) > 0.1)[0]
            X = np.take(X, index_not_constant, axis=0)
            position = np.take(position, index_not_constant)
            chrom = np.take(chrom, index_not_constant)

        X = X.astype('float32')
        # add random noise to the input matrices
        if noise > 0:
            X += np.random.normal(scale=noise, size=X.shape)

        pos = np.vstack([chrom, np.zeros(position.shape[0]), position]).T
        test_snps = SnpData(iid=iid, sid=position.astype('S'),
                            val=X.T, pos=pos)
        del X
    return test_snps

def get_num_snps(genotype_file):
    group = None
    if ':' in genotype_file:
        i = genotype_file.index(':')
        group = genotype_file[(i + 1):]
        genotype_file = genotype_file[:i]
    with h5py.File(genotype_file, 'r') as f:
        if group is None:
            g = f
        else:
            g = f[group]
        n_snps = g['X'].shape[0]
    return n_snps

def test_single_snp(args):
    import fastlmm
    from pysnptools.snpreader import SnpData, Pheno, SnpReader
    from fastlmm.association import single_snp
    from utils import read_hdf5_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import fastlmm.util.util as flutil

    logger.info('read phenotypes from file: ' + args.phenotype_file)
    phenotypes = pd.read_table(args.phenotype_file)
    iid = np.repeat(phenotypes['id'].values.astype('S')[:, np.newaxis], 2, axis=1)
    if args.sample_indices_file is not None:
        logger.info('read indices from file: ' + args.sample_indices_file)
        sample_indices = read_hdf5_dataset(args.sample_indices_file)
    else:
        sample_indices = np.nonzero((phenotypes['type'] == 'training').values)[0]
    logger.info('read SNP file (for test): ' + args.snp_file)
    test_snps = get_snpdata(iid, args.snp_file, sample_indices=sample_indices)
    logger.info('read SNP file (for K0): ' + args.k0_file)
    K0 = get_snpdata(iid, args.k0_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df_pheno = phenotypes[phenotypes['type'] == 'training'].copy()
    df_pheno['fid'] = df_pheno['id']
    df_pheno['iid'] = df_pheno['id']
    traits = ('trait1', 'trait2', 'trait3')
    for trait in traits:
        pheno_file = os.path.join(args.output_dir, 'pheno.%s.txt' % trait)
        logger.info('create Pheno file: ' + pheno_file)
        df_pheno[['fid', 'iid', trait]].to_csv(pheno_file,
                                               index=False, sep='\t',
                                               header=False)
        pheno = Pheno(pheno_file)
        logger.info('run FastLMM for single SNP test for %s'%trait)
        results_df = single_snp(test_snps, pheno, K0=K0, count_A1=True, GB_goal=args.GB_goal)
        result_file = os.path.join(args.output_dir, 'single_snp.' + trait)
        logger.info('save results to file: ' + result_file)
        results_df.to_hdf(result_file, trait)

        if args.manhattan:
            plot_file = os.path.join(args.output_dir, 'manhattan.%s.pdf'%trait)
            logger.info('create Manhattan plot: ' + plot_file)
            plt.clf()
            flutil.manhattan_plot(results_df.as_matrix(["Chr", "ChrPos", "PValue"]),
                                  pvalue_line=1e-5,
                                  xaxis_unit_bp=False)
            plt.savefig(plot_file)

def run_fastlmm(args):
    from pysnptools.snpreader import SnpData, Pheno, SnpReader
    from utils import prepare_output_file, read_cvindex
    from fastlmm.inference import FastLMM
    import dill as pickle

    logger.info('read phenotypes from file: ' + args.phenotype_file)
    phenotypes = pd.read_table(args.phenotype_file)
    iid = np.repeat(phenotypes['id'].values.astype('S')[:, np.newaxis], 2, axis=1)
    if args.cvindex_file is not None:
        logger.info('read indices from file: ' + args.cvindex_file)
        train_index, test_index = read_cvindex(args.cvindex_file)
    else:
        train_index = np.nonzero((phenotypes['type'] == 'training').values)[0]
        test_index = np.nonzero((phenotypes['type'] == 'test').values)[0]

    n_snps_total = get_num_snps(args.snp_file)
    n_snps_sel = min(n_snps_total, args.n_snps)
    logger.info('number of sampled SNPs: %d' % n_snps_sel)
    sel_snps = np.random.choice(n_snps_total, size=n_snps_sel)

    logger.info('read SNP file (for test): ' + args.snp_file)
    test_snps = get_snpdata(iid, args.snp_file,
                            transpose=args.transpose_x,
                            snp_indices=sel_snps, std_filter_indices=train_index)
    logger.info('number of sampled SNPs after filtering by std: %d'%test_snps.shape[1])
    logger.info('read SNP file (for K0): ' + args.k0_file)
    K0 = get_snpdata(iid, args.k0_file, transpose=args.transpose_k0)

    if args.seed:
        logger.info('set random seed for numpy: %d'%args.seed)
        np.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df_pheno = phenotypes.copy()
    df_pheno['fid'] = df_pheno['id']
    df_pheno['iid'] = df_pheno['id']
    traits = ('trait1', 'trait2', 'trait3')
    for trait in traits:
        pheno_file = os.path.join(args.output_dir, 'pheno.%s.txt' % trait)
        logger.info('create Pheno file: ' + pheno_file)
        df_pheno.loc[train_index, ['fid', 'iid', trait]].to_csv(pheno_file,
                                               index=False, sep='\t',
                                               header=False)
        pheno = Pheno(pheno_file)
        logger.info('train FastLMM model for %s'%trait)
        model = FastLMM(GB_goal=args.GB_goal, force_low_rank=True)
        model.fit(X=test_snps[train_index, :], y=pheno, K0_train=K0,
                  penalty=args.penalty, Smin=1.0)
        logger.info('fitted h2: %f'%model.h2raw)
        logger.info('predict using the FastLMM model for %s' % trait)
        y_mean, y_var = model.predict(X=test_snps[test_index, :],
                                      K0_whole_test=K0[test_index, :])
        y_true = phenotypes[trait][test_index].values
        result_file = os.path.join(args.output_dir, 'predictions.%s'%trait)
        logger.info('save predictions to file: ' + result_file)
        prepare_output_file(result_file)
        with h5py.File(result_file, 'w') as f:
            f.create_dataset('y_mean', data=y_mean.val)
            f.create_dataset('y_var', data=y_var.val)
            f.create_dataset('y_true', data=y_true)
            f.create_dataset('h2raw', data=model.h2raw)
            f.create_dataset('sel_snps', data=sel_snps)

        model_file = os.path.join(args.output_dir, 'model.fastlmm.%s'%trait)
        logger.info('save model to file: ' + model_file)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    main_parser = argparse.ArgumentParser(description='Preprocessing script')
    subparsers = main_parser.add_subparsers(dest='command')

    parser = subparsers.add_parser('single_snp',
                                   help='GWAS analysis of single SNPs')
    parser.add_argument('-i', '--snp-file', type=str, required=True,
                        help='file containing SNPs to test')
    parser.add_argument('--sample-indices-file', type=str,
                        help='only use samples defined in the indices file')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='phenotype file in text format (pheno_emaize.txt)')
    parser.add_argument('--k0-file', type=str, required=True,
                        help='file containing SNPs to construct the genetic similarity matrix')
    parser.add_argument('--manhattan', action='store_true',
                        help='create a manhattan plot')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='output directory in HDF5 format containing the DataFrames')
    parser.add_argument('--GB-goal', type=int, default=8,
                        help='gigabytes of memory the run should use')

    parser = subparsers.add_parser('fastlmm',
                                   help='train the FastLMM model')
    parser.add_argument('-i', '--snp-file', type=str, required=True,
                        help='file containing SNPs to test')
    parser.add_argument('--cvindex-file', type=str,
                        help='file generated by create_cv_folds.py (filename:<cv_fold>')
    parser.add_argument('--phenotype-file', type=str, required=True,
                        help='phenotype file in text format (pheno_emaize.txt)')
    parser.add_argument('--k0-file', type=str, required=True,
                        help='file containing SNPs to construct the genetic similarity matrix')
    parser.add_argument('--transpose-k0', action='store_true', default=False,
                        help='transpose the K0 matrix such that rows are features')
    parser.add_argument('--transpose-x', action='store_true', default=False,
                        help='transpose the SNP matrix such that rows are features')
    parser.add_argument('--n-snps', type=int, default=500,
                        help='maximum number of SNPs to sample')
    parser.add_argument('--seed', type=int, help='random seed for numpy')
    parser.add_argument('--penalty', type=float, default=0.0,
                        help='regularization term for the weights of fixed effects')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='output directory in HDF5 format containing the DataFrames')
    parser.add_argument('--GB-goal', type=int, default=8,
                        help='gigabytes of memory the run should use')

    args = main_parser.parse_args()


    logger = logging.getLogger('run_fastlmm.' + args.command)

    import pandas as pd
    import numpy as np
    import h5py

    if args.command == 'single_snp':
        test_single_snp(args)
    elif args.command == 'fastlmm':
        run_fastlmm(args)