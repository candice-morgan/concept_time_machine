import os
import json
from glob import glob
from optparse import OptionParser
from collections import defaultdict, Counter

from tqdm import tqdm


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/coha/',
                      help='Base directory: default=%default')

    (options, args) = parser.parse_args()

    cofea_dir = options.basedir

    tokenized_dir = os.path.join(cofea_dir, 'tokenized')
    split_dir = os.path.join(cofea_dir, 'splits')
    counts_dir = os.path.join(cofea_dir, 'counts')

    if not os.path.exists(counts_dir):
        os.makedirs(counts_dir)

    files = sorted(glob(os.path.join(tokenized_dir, '*.jsonlist')))
    for infile in files:
        print(infile)
        basename = os.path.basename(infile).split('.')[0]
        split_file = os.path.join(split_dir, basename + '_train_test_split.json')
        with open(split_file) as f:
            split = json.load(f)
        train_ids = set(split['train'])
        test_ids = set(split['test'])

        train_token_counts = Counter()
        test_token_counts = Counter()

        with open(infile) as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            tokens = line['tokens']
            line_id = line['id']
            if line_id in train_ids:
                train_token_counts.update(tokens)
            elif line_id in test_ids:
                test_token_counts.update(tokens)

        with open(os.path.join(counts_dir, basename + '_train_token_counts_all.json'), 'w') as fo:
            json.dump(train_token_counts, fo, indent=2)

        with open(os.path.join(counts_dir, basename + '_test_token_counts_all.json'), 'w') as fo:
            json.dump(test_token_counts, fo, indent=2)



if __name__ == '__main__':
    main()
