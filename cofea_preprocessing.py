import os
import re
import json
from glob import glob
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("--input_dir",default=os.path.join('data', 'cofea_full'))
argp.add_argument("--output_dir",default=os.path.join('data','cofea_processed'))
args = argp.parse_args()

# load the data and fix some errors
indir = args.input_dir
outdir = args.output_dir

files = [os.path.join(indir, f) for f in os.listdir(indir) if os.path.isfile(os.path.join(indir, f)) and ".json" in f]
docs = []
for infile in files:
    basename = os.path.basename(infile)
    with open(infile) as f:
        data = json.load(f)
    print(infile, len(data), type(data))
    for d in data:
        # fix one document with many problems (http://founders.archives.gov/documents/Jefferson/99-01-02-9951)
        if d['id'] == 'fndrs.jefferson.99-01-02-9951':
            d['author'] == 'Wright, Robert'
            d['year'] = 1809
            d['decade'] = 1800
            d['collection'] = 'Jefferson Papers'
        # fix another that has year and decade listed as 2000:
        elif d['id'] == 'fndrs.jefferson.01-42-02-0442-0002':
            d['year'] = 1804
            d['decade'] = 1800
        # fix one document that clearly has the wrong year/decade (17626/17606)
        elif d['id'] == 'evans.N07112':
            d['year'] = 1762
            d['decade'] = 1760
        # fix years and decades for Elliot's debates (many listed as "2018")
        elif d['source'] == "Elliot's Debates":
            if 'year' in d and int(d['year']) == 2018:
                d.pop('year')
            d['decade'] = 1780
        # convert all years and decades to ints
        if 'year' in d:
            d['year'] = int(d['year'])
        if 'decade' in d:
            d['decade'] = int(d['decade'])
        if d['title'] != 'Editorial Note':
            docs.append(d)

outlines = []
for d in docs:
    if 'source' in d and 'year' in d and 'decade' in d:
        outlines.append({'id': d['id'],
                         'decade': d['decade'],
                         'year': int(d['year']),
                         'source': d['source'],
                         'text': d['body'],})
if len(outlines)> 0:
    with open(os.path.join(outdir,'cofea.jsonlist'), 'w') as f:
            for line in outlines:
                f.write(json.dumps(line) + '\n')