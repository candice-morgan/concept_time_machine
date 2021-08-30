import os
import re
import sys
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/coha/',
                      help='Dat adir: default=%default')
    parser.add_option('--emb-dir', type=str, default='data/coha/bert-base-uncased_test/',
                      help='Embedding dir: default=%default')
    parser.add_option('--target-file', type=str, default='coha/targets.json',
                      help='Targets file: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    emb_dir = options.emb_dir
    target_file = options.target_file

    with open(target_file) as f:
        target_words = set(json.load(f))

    tokenized_dir = os.path.join(basedir, 'tokenized')

    files = sorted(glob(os.path.join(tokenized_dir, '*.jsonlist')))

    for infile in files:
        basename = os.path.splitext(os.path.basename(infile))[0]

        lefts = []
        rights = []
        meta_file = os.path.join(emb_dir, basename + '_token_info.json')
        token_file = os.path.join(tokenized_dir, basename + '.jsonlist')
        with open(meta_file) as f:
            metadata = json.load(f)
        segments = metadata['segments']
        tokens = metadata['tokens']
        token_indices = metadata['token_indices']

        print(infile, len(segments))

        tokenized = {}
        with open(token_file) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            tokenized[line['id']] = line['tokens']

        for i, segment in enumerate(segments):
            token_index = token_indices[i]
            token = tokens[i]
            try:
                assert tokenized[segment][token_index] == token

            except AssertionError as e:
                print(segment)
                print(token)
                print(token_index)
                print(tokenized[segment])
                print(tokenized[segment][token_index])
                raise e

            lefts.append(' '.join(tokenized[segment][max(0, token_index-10):token_index]))
            rights.append(' '.join(tokenized[segment][token_index+1:token_index+11]))

        with open(os.path.join(emb_dir, basename + '_contexts.json'), 'w') as f:
            json.dump({'lefts': lefts,
                       'rights': rights}, f, indent=2)


if __name__ == '__main__':
    main()
