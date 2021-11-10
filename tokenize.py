import json
import os
import json
from glob import glob
from tqdm import tqdm
from transformers import BertTokenizer
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("--input_dir", default=os.path.join('data', 'cofea_processed'))
argp.add_argument("--output_dir", default=os.path.join('data','cofea_processed'))
args = argp.parse_args()

# load the data and fix some errors
indir = args.input_dir
outdir = args.output_dir

files = sorted(glob(os.path.join(indir, '*.jsonlist')))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for infile in files:
    basename = os.path.basename(infile).sub('.jsonlist',"")+'_tokenized.jsonlist'
    doc_id = 0
    with open(infile) as f:
        docs = f.readlines()

    outlines = []
    for doc in tqdm(docs):
        doc = json.loads(doc)
        doc_id = doc['id']
        text = doc['text']
        # convert to tokens using BERT
        raw_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(text, add_special_tokens=False)]
        # rejoin into concatenated words
        rejoined_pieces = []
        if len(raw_pieces) > 0:
            for p_i, piece in enumerate(raw_pieces):
                if p_i == 0:
                    rejoined_pieces.append(piece)
                elif piece.startswith('##'):
                    rejoined_pieces[-1] += piece
                else:
                    rejoined_pieces.append(piece)
            text = [x.replace('##', "") for x in raw_pieces]
            text = ' '.join(text)
            outlines.append({'doc_id': doc_id, 'text': text, 'tokens': rejoined_pieces})

    outfile = os.path.join(outdir, basename)
    with open(outfile, 'w') as fo:
        for line in outlines:
            fo.write(json.dumps(line) + '\n')