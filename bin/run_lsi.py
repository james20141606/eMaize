#! /usr/bin/env python
import logging
import argparse, sys, os, errno
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('run_lsi')

def get_chunks(corpus, chunksize=1):
    n = corpus.shape[1]
    for i in range(n/chunksize):
        end = min(n, (i + 1)*chunksize)
        yield gensim.matutils.Dense2Corpus(corpus[:, (i*chunksize):end])

def prepare_output_file(filename):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genotype-file', '-i', type=str, required=True)
    parser.add_argument('--phenotype-file', type=str, required=True)
    parser.add_argument('--sample-names-file', type=str, required=True)
    parser.add_argument('--model-file', '-o', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=0,
        help='number of samples to process each time, 0 for all samples')
    parser.add_argument('--rank', '-r', type=int, default=1000)
    args = parser.parse_args()

    import gensim
    import h5py
    import numpy as np
    import pandas as pd

    logger.info('read phenotypes from ' + args.phenotype_file)
    phenotypes = pd.read_table(args.phenotype_file)
    phenotypes.index = phenotypes['id']
    n_samples = phenotypes.shape[0]

    logger.info('read genotypes from ' + args.genotype_file)
    with open(args.genotype_file, 'rb') as f:
        genotypes = np.frombuffer(f.read(), dtype='int8').reshape((-1, n_samples))

    logger.info('total number of samples: %d'%phenotypes.shape[0])
    logger.info('read sample names from ' + args.sample_names_file)
    with open(args.sample_names_file, 'r') as f:
        sample_names = f.read().strip().split()
    logger.info('selected number of samples: %d'%len(sample_names))

    selected_index = np.nonzero(phenotypes['id'].isin(sample_names).values)[0]
    genotypes = np.take(genotypes, selected_index, axis=1)

    logger.info('run LsiModel with chunksize=%d, rank=%d'%(args.chunk_size, args.rank))
    if args.chunk_size > 0:
        chunks = get_chunks(genotypes, args.chunk_size)
        model = gensim.models.LsiModel(next(chunks), num_topics=args.rank, chunksize=args.chunk_size, distributed=True)
        for chunk in chunks:
            model.add_documents(chunk, chunksize=args.chunk_size)
    else:
        genotypes = gensim.matutils.Dense2Corpus(genotypes)
        model = gensim.models.LsiModel(genotypes, num_topics=args.rank, distributed=True)
    prepare_output_file(args.model_file)
    model.save(args.model_file)
