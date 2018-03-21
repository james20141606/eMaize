#! /usr/bin/env python
import numpy as np
import sys

def load_corpus_from_genotype(sample_id):
    snp_id = 0
    genotypes = []
    for line in sys.stdin:
        if line.startswith('snp'):
            continue
        c = line.strip().split('\t')
        a = c[1].split('/')
        genotype = c[sample_id + 4]
        if genotype == a[0] + a[0]:
            genotypes.append(1)
        else:
            genotypes.append(0)
        if genotype == a[0] + a[1]:
            genotypes.append(1)
        else:
            genotypes.append(0)
        if genotype == a[1] + a[1]:
            genotypes.append(1)
        else:
            genotypes.append(0)
    genotypes = np.asarray(genotypes, dtype='int8')
    print genotypes[:100]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >>sys.stderr, 'Usage: {} sample_id'.format(sys.argv[0])
        sys.exit(1)
    try:
        load_corpus_from_genotype(int(sys.argv[1]))
    except KeyboardInterrupt:
        pass
