import os
import re
import sys
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
#from scipy.special import log_softmax


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='data/coha/',
                      help='Base directory: default=%default')
    parser.add_option('--target-file', type=str, default='coha/targets.json',
                      help='.json file with list of target words: default=%default')
    parser.add_option('--model-type', type=str, default='bert',
                      help='Model type: default=%default')
    parser.add_option('--model', type=str, default='bert-base-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--tokenizer', type=str, default=None,
                      help='Tokenizer name or path (defaults to model): default=%default')
    parser.add_option('--batch-size', type=int, default=200,
                      help='Batch size: default=%default')
    parser.add_option('--layers', type=str, default='11',
                      help='Comma-separated list of layers to save: default=%default')
    parser.add_option('--min-count', type=int, default=1000,
                      help='Min count (over all years): default=%default')
    parser.add_option('--device', type=int, default=0,
                      help='GPU to use: default=%default')
    parser.add_option('--seed', type=int, default=54,
                      help='Random seed: default=%default')
    parser.add_option('--debug', action="store_true", default=False,
                      help='Debug: default=%default')
    parser.add_option('--train', action="store_true", default=False,
                      help='Use train instead of test data: default=%default')

    (options, args) = parser.parse_args()

    cofea_dir = options.basedir
    target_file = options.target_file
    model_type = options.model_type
    model_name_or_path = options.model
    tokenizer = options.tokenizer
    batch_size = options.batch_size
    layers = options.layers
    min_count = options.min_count
    device = options.device
    seed = options.seed
    debug = options.debug
    use_train = options.train

    np.random.seed(seed)

    layers = [int(layer) for layer in layers.split(',')]
    print('layers:', layers)

    with open(target_file) as f:
        target_words = set(json.load(f))

    tokenized_dir = os.path.join(cofea_dir, 'tokenized')
    split_dir = os.path.join(cofea_dir, 'splits')
    counts_dir = os.path.join(cofea_dir, 'counts')

    files = sorted(glob(os.path.join(tokenized_dir, '*.jsonlist')))
    outdir = os.path.join(cofea_dir, model_name_or_path)
    if use_train:
        outdir += '_train'
    else:
        outdir += '_test'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading model")
    if model_type == 'bert':
        model_class = BertModel
        tokenizer_class = BertTokenizer
    elif model_type == 'roberta':
        model_class = RobertaModel
        tokenizer_class = RobertaTokenizer
    else:
        raise ValueError("Model type not recognized")

    # Load pretrained model/tokenizer
    if tokenizer is None:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    else:
        tokenizer = tokenizer_class.from_pretrained(tokenizer)

    model = model_class.from_pretrained(model_name_or_path)

    # move the model to the GPU
    torch.cuda.set_device(device)
    device = torch.device("cuda", device)
    model.to(device)

    for infile in files:
        basename = os.path.basename(infile)
        split_file = os.path.join(split_dir, basename.split('.')[0] + '_train_test_split.json')
        with open(split_file) as f:
            splits = json.load(f)
        train_set = splits['train']
        test_set = splits['test']

        if use_train:
            target_ids = set(train_set)
            counts_file = os.path.join(counts_dir, basename.split('.')[0] + '_train_token_counts_all.json')
        else:
            target_ids = set(test_set)
            counts_file = os.path.join(counts_dir, basename.split('.')[0] + '_test_token_counts_all.json')
        with open(counts_file) as f:
            token_counts = Counter(json.load(f))

        for token in sorted(target_words):
            print(token, token_counts[token])

        print(len(token_counts))
        token_probs = defaultdict(float)
        for token, count in token_counts.items():
            if re.match(r'.*[a-zA-Z0-9].*', token) is not None:
                if token in target_words and count <= min_count:
                    expected = count
                elif token in target_words:
                    expected = subsampling_function(count, min_n=min_count) * 10
                else:
                    expected = subsampling_function(count, min_n=min_count)
                token_probs[token] = expected / count
        print(len(token_probs))

        token_probs['[CLS]'] = 0
        token_probs['[SEP]'] = 0

        # Just put the output into some lists for now
        segment_id_list = []
        token_list = []
        piece_index_list = []
        token_index_list = []
        vectors_by_layer = defaultdict(list)

        with open(infile) as f:
            lines = f.readlines()

        lines = [json.loads(line) for line in lines]
        lines = [line for line in lines if line['id'] in target_ids]
        n_lines = len(lines)
        print(n_lines, 'lines')

        to_encode = []
        encoded = []
        targets = []
        token_indices = []
        lengths = []
        line_ids = []
        if debug:
            max_lines = batch_size * 10
        else:
            max_lines = len(lines)
        for line_index, line in enumerate(tqdm(lines[:max_lines])):
            line_id = line['id']
            text = line['span']
            start_indices = []
            token_lengths = []
            tokens = []
            pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(text, add_special_tokens=True, truncation=True)]
            for p_i, piece in enumerate(pieces):
                if p_i == 0:
                    start_indices.append(p_i)
                    token_lengths.append(1)
                    tokens.append(piece)
                elif piece.startswith('##'):
                    token_lengths[-1] += 1
                    tokens[-1] += piece
                else:
                    start_indices.append(p_i)
                    token_lengths.append(1)
                    tokens.append(piece)
            if debug or line_id == 'fic_1811_8990_00000':
                print(tokens)
                print(token_lengths)

            sample = []
            if len(tokens) > 2:
                for t_i, token in enumerate(tokens):
                    if token_probs[token] > 0:
                        if np.random.rand() < token_probs[token]:
                            sample.append(t_i)

            if len(sample) > 0:
                to_encode.append(text)
                encoded.append([tokens[i] for i in sample])
                # save the targets in token pieces
                targets.append([start_indices[i] for i in sample])
                # save the targets token indices
                token_indices.append(sample)
                lengths.append([token_lengths[i] for i in sample])
                line_ids.append(line_id)
                if debug or line_id == 'fic_1811_8990_00000':
                    print(sample)
                    print([tokens[i] for i in sample])
                    print([start_indices[i] for i in sample])
                    print([token_lengths[i] for i in sample])

            if len(to_encode) == batch_size or (line_index == (n_lines-1) and len(to_encode) > 1):

                if debug or line_id == 'fic_1811_8990_00000':
                    print(to_encode)
                    print(encoded)
                    print(targets)
                    print(token_indices)
                    print(lengths)
                    print(line_ids)

                output = tokenizer(to_encode,
                                   add_special_tokens=True,
                                   padding=True,
                                   truncation=True,
                                   return_tensors='pt')
                input_ids = output['input_ids']
                n_rows, n_tokens = input_ids.shape
                #print(n_rows, n_tokens)
                attention_mask = output['attention_mask']

                if debug:
                    print(n_rows)
                    pieces = []
                    for row in range(n_rows):
                        pieces.append([tokenizer.ids_to_tokens[int(i)] for i in input_ids[row, :]])

                input_ids_on_device = input_ids.to(device)
                attention_mask_on_device = attention_mask.to(device)

                # process the text through the model
                with torch.no_grad():
                    try:
                        output = model(input_ids=input_ids_on_device,
                                       attention_mask=attention_mask_on_device,
                                       output_hidden_states=True)
                        hidden_states = output[2]

                        vectors_np = {layer: hidden_states[layer].detach().cpu().numpy() for layer in layers}

                        for row in range(n_rows):
                            for t_i, target in enumerate(targets[row]):
                                segment_id_list.append(line_ids[row])
                                token_list.append(encoded[row][t_i])
                                piece_index_list.append(target)
                                # subtract one to account for [CLS] token
                                token_index_list.append(token_indices[row][t_i]-1)
                                for layer in layers:
                                    vectors_by_layer[layer].append(np.array(vectors_np[layer][row, target, :].copy(), dtype=np.float32))

                                if debug:
                                    print("processing target in row")
                                    print(row)
                                    print(line_ids[row])
                                    print(target)
                                    print(t_i)
                                    print(encoded[row][t_i])
                                    print(pieces[row][target])
                                    print(layers[0])
                                    print(vectors_np[layers[0]].shape)
                                    print(vectors_np[layers[0]][row, target, :].shape)

                    except Exception as e:
                        print(n_rows)
                        print(encoded)
                        raise e

                to_encode = []
                encoded = []
                targets = []
                token_indices = []
                lengths = []
                line_ids = []

        print("Concatenating and saving data")
        for layer in layers:
            stacked_vectors = np.vstack(vectors_by_layer[layer])
            np.savez_compressed(os.path.join(outdir, basename.split('.')[0] + '_layer_{:d}_vectors.npz'.format(layer)), vectors=stacked_vectors)

        with open(os.path.join(outdir, basename.split('.')[0] + '_token_info.json'), 'w') as f:
            json.dump({'segments': segment_id_list,
                       'tokens': token_list,
                       'token_indices': token_index_list,
                       'piece_indices': piece_index_list}, f, indent=2)


def subsampling_function(n, min_n=1000, factor=14.48):
    if n < min_n:
        return 0
    else:
        return np.log(n) * factor


if __name__ == '__main__':
    main()
