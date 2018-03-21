#! /usr/bin/env python
import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('genotype_to_avinput')

def read_fasta(filename):
    def append_extra_line(f):
        """Yield an empty line after the last line in the file
        """
        for line in f:
            yield line
        yield ''
    with open(filename, 'r') as f:
        name = None
        seq = ''
        for line in append_extra_line(f):
            if line.startswith('>') or (len(line) == 0):
                if (len(seq) > 0) and (name is not None):
                    yield (name, seq)
                if line.startswith('>'):
                    name = line.strip()[1:].split()[0]
                    seq = ''
            else:
                if name is None:
                    raise ValueError('the first line does not start with ">"')
                seq += line.strip()

def genotype_to_avinput(genome_file):
    logger.info('load genome sequences')
    genome = dict(read_fasta(genome_file))
    logger.info('read genotype')
    sys.stdin.readline()
    for line in sys.stdin:
        c = line.split('\t')
        alleles = c[1].split('/')
        pos = int(c[3])
        ref = genome[c[2]][pos - 1]
        for allele in alleles:
            if allele != ref:
                sys.stdout.write('\t'.join((c[2], c[3], c[3], ref, allele, 'comments: ' + c[0])))
                sys.stdout.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >>sys.stderr, 'Usage: {} genome_file'.format(sys.argv[0])
        sys.exit(1)
    genome_file = sys.argv[1]

    try:
        genotype_to_avinput(genome_file)
    except KeyboardInterrupt:
        pass
