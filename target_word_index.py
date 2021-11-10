import json
import os
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer
import pickle
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("--input_dir", default=os.path.join('data', 'cofea_processed'))
argp.add_argument("--output_dir", default=os.path.join('data','cofea_processed'))
argp.add_argument("--target_file", default=os.path.join('data','constitution_words.txt'))
args = argp.parse_args()

# load the data and fix some errors
indir = args.input_dir
outdir = args.output_dir
word_file = args.target_file

files = sorted(glob(os.path.join(indir, '*_tokenized.jsonlist')))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenize key words and phrases
with open(word_file, 'r',encoding = 'utf-8') as f:
    target_words = f.read().splitlines()
target_words = [tokenizer.tokenize(x) for x in target_words]
target_words_cleaned = []

for x in target_words:
    # rejoin into concatenated words
    rejoined_pieces = []
    for p_i, piece in enumerate(x):
        if p_i == 0:
            rejoined_pieces.append(piece)
        elif piece.startswith('##'):
            rejoined_pieces[-1] += piece
        else:
            rejoined_pieces.append(piece)
    target_words_cleaned.append(rejoined_pieces)

# collect index of tokens in the documents
index = defaultdict(set)

for i, file in enumerate(files):
    with open(file) as f:
        docs = f.readlines()
    for x, doc in enumerate(tqdm(docs)):
        doc = json.loads(doc)
        tokens = doc['tokens']
        doc_index = 0
        token = []
        token_index = 0
        while doc_index < len(tokens):
            if doc_index == 0:
                token = tokens[doc_index]
            elif tokens[doc_index].startswith('##'):
                token += tokens[doc_index]
            else:
                # save the current token and its start index
                index[token].add((i, x, token_index))
                # start a new token
                token = tokens[doc_index]
                token_index = doc_index
            # save the final token if we are at the end
            if doc_index == len(tokens) - 1:
                index[token].add((i, x, token_index))

            # update the index based on the broken up tokens, not the whole word
            doc_index += 1

# collect only the target words and phrase indexes and save
target_index = {}
for word in tqdm(target_words_cleaned):
    if len(word) == 1:
        # just one word
        target_index[word[0]]=index.get(word[0],set())

    else:
        # we have a phrase
        phrase_indexes = []
        start = index.get(word[0])
        for f_id,doc_id,doc_index in start:
            match = False
            for x,piece in enumerate(word[1:]):
                if (f_id,doc_id,doc_index+x+1) in index.get(piece,set()):
                    match = True
                else:
                    match = False
                    break

            if match:
                phrase_indexes.append((f_id,doc_id,doc_index))
        target_index[' '.join(word)] = set(phrase_indexes)

with open(os.path.join(outdir,'target_word_index.dict'),'wb') as f:
    pickle.dump(target_index,file=f)