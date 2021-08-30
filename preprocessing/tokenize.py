import os
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer


# Tokenize COHA with BERT

def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/coha/',
                      help='Directory with processed dir: default=%default')
    parser.add_option('--max-len', type=int, default=256,
                      help='Max # of word tokens per span: default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--model', type=str, default='bert-base-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    indir = os.path.join(basedir, 'processed')
    outdir = os.path.join(basedir, 'tokenized')
    max_len = options.max_len
    model_type = options.model_type
    model_name_or_path = options.model
    tokenizer = options.tokenizer

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading model")
    if model_type == 'bert':
        tokenizer_class = BertTokenizer
    elif model_type == 'roberta':
        tokenizer_class = RobertaTokenizer
    else:
        raise ValueError("Model type not recognized")

    # Load pretrained model/tokenizer
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    files = sorted(glob(os.path.join(indir, '*.jsonlist')))
    for infile in files:
        basename = os.path.basename(infile)
        print(infile)
        with open(infile) as f:
            lines = f.readlines()

        outlines = []
        max_n_spans = 0
        max_n_pieces = 0
        gt_512 = 0
        for line in tqdm(lines):
            line = json.loads(line)
            line_id = line['id']
            text = line['text']
            all_tokens = text.split()

            n_tokens = len(all_tokens)

            if n_tokens > 0:
                n_spans = int(np.ceil(n_tokens / max_len))
                span_length = int(n_tokens / n_spans)
                max_n_spans = max(max_n_spans, n_spans)

                for s_i in range(n_spans):
                    if s_i < (n_spans - 1):
                        span_tokens = all_tokens[s_i * span_length: (s_i + 1) * span_length]
                    else:
                        span_tokens = all_tokens[s_i * span_length:]
                    span = ' '.join(span_tokens)

                    span_id = line_id + '_' + str(s_i).zfill(5)

                    # convert to tokens using BERT
                    raw_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(span, add_special_tokens=False)]
                    max_n_pieces = max(max_n_pieces, len(raw_pieces))
                    if len(raw_pieces) > 512:
                        gt_512 += 1

                    rejoined_pieces = []
                    # rejoin into concatenated words
                    if len(raw_pieces) > 0:
                        for p_i, piece in enumerate(raw_pieces):
                            if p_i == 0:
                                rejoined_pieces.append(piece)
                            elif piece.startswith('##'):
                                rejoined_pieces[-1] += piece
                            else:
                                rejoined_pieces.append(piece)
                        outlines.append({'id': span_id, 'doc_id': line_id, 'span': span, 'tokens': rejoined_pieces})

        print(infile, max_n_spans, max_n_pieces, len(lines), gt_512)
        outfile = os.path.join(outdir, basename)
        with open(outfile, 'w') as fo:
            for line in outlines:
                fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
