#! /usr/bin/env python
import gensim
import h5py
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_chunks(corpus, chunksize=1):
    n = corpus.shape[1]
    for i in range(n/chunksize):
        end = min(n, (i + 1)*chunksize)
        yield gensim.matutils.Dense2Corpus(corpus[:, (i*chunksize):end])

f = h5py.File('data/corpus/random.h5')
corpus = f['corpus'][:]
f.close()
chunksize = 50
chunks = get_chunks(corpus, chunksize)
model = gensim.models.LsiModel(next(chunks), num_topics=200, chunksize=chunksize, distributed=True)
for chunk in chunks:
    model.add_documents(chunk, chunksize=chunksize)
model.save('models/random.lsi_model')
