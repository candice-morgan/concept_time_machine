import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/coha/',
                      help='Base directory: default=%default')
    parser.add_option('--train-prop', type=float, default=0.5,
                      help='Prop to use for training: default=%default')
    parser.add_option('--seed', type=int, default=54,
                      help='Random seed: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    train_prop = options.train_prop
    seed = options.seed

    np.random.seed(seed)

    train_set = set()
    test_set = set()

    indir = os.path.join(basedir, 'tokenized')
    outdir = os.path.join(basedir, 'splits')
    files = sorted(glob(os.path.join(indir, '*.jsonlist')))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for infile in files:
        print(infile)
        basename = os.path.basename(infile).split('.')[0]

        with open(infile) as f:
            lines = f.readlines()

        line_ids_by_doc_id = defaultdict(set)
        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            doc_id = '_'.join(line_id.split('_')[:-1])

            # get all the segment ids for each document
            line_ids_by_doc_id[doc_id].add(line_id)

        # randomly split the documents
        doc_ids = sorted(line_ids_by_doc_id)
        np.random.shuffle(doc_ids)
        for i, doc_id in enumerate(doc_ids):
            if np.random.rand() < train_prop:
                train_set.update(line_ids_by_doc_id[doc_id])
            else:
                test_set.update(line_ids_by_doc_id[doc_id])

        print(basename, len(train_set), len(test_set))

        with open(os.path.join(outdir, basename + '_train_test_split.json'), 'w') as fo:
            json.dump({'train': sorted(train_set), 'test': sorted(test_set)}, fo, indent=2)


if __name__ == '__main__':
    main()
