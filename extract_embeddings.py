import os
import re
import sys
import json
from glob import glob
from collections import Counter, defaultdict
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertModel,BertTokenizer

in_dir = os.path.join('data','preprocessed')
sample_file = os.path.join('data','preprocessed','sample_target_index.dict')
outdir = os.path.join('data','preprocessed')
out_file = os.path.join('data','preprocessed','cofea_sampled_vectors')
batch_size = 200
layers = '10,11'

# collect index of tokens in the documents
files = sorted(glob(os.path.join(in_dir, '*_tokenized.jsonlist')))
docs = []
for file in files:
    with open(file) as f:
        docs.append(f.readlines())
        
# get sample index
with open(sample_file,'rb') as f:
    sample_index = pickle.load(f)
    
#layers
layers = [int(layer) for layer in layers.split(',')]

# load the model
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# move the model to the GPU
device = 'cuda'
if torch.cuda.is_available():
    model.to(device)

def get_context(token_ids, target_position, sequence_length=128):
    """
    Given a text containing a target word, return the sentence snippet which surrounds the target word
    (and the target word's position in the snippet).
    :param token_ids: list of token ids (for an entire line of text)
    :param target_position: index of the target word's position in `tokens`
    :param sequence_length: desired length for output sequence (e.g. 128, 256, 512)
    :return: (context_ids, new_target_position)
                context_ids: list of token ids for the output sequence
                new_target_position: index of the target word's position in `context_ids`
    """
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position

# Just put the output into some lists for now
token_index_list = []
to_encode = []
vectors_by_layer = defaultdict(dict)
for layer in layers:
    vectors_by_layer[layer] = defaultdict(list)
    
for target_word in tqdm(sample_index):
    for line_index,example_info in enumerate(sample_index[target_word]):
        file_id,doc_id,index = example_info
        doc = json.loads(docs[file_id][doc_id])
        tokens = doc['tokens']
        # now we get the context
        context_ids, pos_in_context = get_context(tokens, index)
        input_ids = tokenizer.convert_tokens_to_ids(
            ['[CLS]']+context_ids+['[SEP]'])
        to_encode.append(input_ids)
        token_index_list.append(pos_in_context+1) #increment because we add CLS
        # we reached the batch limit and wil, now extract BERT embeddings
        if len(to_encode) == batch_size or (line_index == (len(sample_index[target_word])-1) and len(to_encode)>1):
            input_tensors = torch.tensor(to_encode)
            input_tensors = input_tensors.to(device)
            n_rows, n_tokens = input_tensors.shape
            with torch.no_grad():
                try:
                    # run usages through language model
                    outputs = model(input_tensors,output_hidden_states=True)
                    hidden_states = outputs[2]
                    vectors_np = {layer: hidden_states[layer].detach().cpu().numpy() for layer in layers}
                    # save the first token of the target word in each example
                    for row in np.arange(len(token_index_list)):
                        pos = token_index_list[row]
                        for layer in layers:
                            vectors_by_layer[layer][target_word].append(
                                np.array(vectors_np[layer][row, pos, :].copy(), dtype=np.float32))                        
                        
                except Exception as e:
                        print(len(to_encode))
                        raise e
            
            
            to_encode = []
            token_index_list = []

for layer in layers:
    with open(out_file+'_layer_'+str(layer)+'.dict','wb') as f:
        pickle.dump(vectors_by_layer[layer],file=f)