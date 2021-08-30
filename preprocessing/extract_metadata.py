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
                      help='Directory with processed dir: default=%default')
    parser.add_option('--fields', type=str, default='source,decade',
                      help='Comma-separated list of fields: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    indir = os.path.join(basedir, 'processed')
    outdir = os.path.join(basedir, 'metadata')
    fields = options.fields

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if fields is not None:
        fields = fields.split(',')
        files = sorted(glob(os.path.join(indir, '*.jsonlist')))
        for infile in files:
            basename = os.path.splitext(os.path.basename(infile))[0]
            metadata = {}
            print(infile)
            with open(infile) as f:
                lines = f.readlines()

            for line in tqdm(lines):
                line = json.loads(line)
                line_id = line['id']
                metadata[line_id] = {}
                for field in fields:
                    try:
                        metadata[line_id][field] = line[field]
                    except KeyError as e:
                        print(line)
                        raise e

            with open(os.path.join(outdir, basename + '.json'), 'w') as f:
                json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
