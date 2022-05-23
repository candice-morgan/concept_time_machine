import gensim,logging
from gensim.models.word2vec import Word2Vec
import os
import sys
from os import listdir,makedirs
from os.path import isfile, join,exists
import glob
import json
import spacy
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict,Counter
from nltk.corpus import stopwords
import string
import nltk
import numpy as np
import codecs
import random
from tqdm import tqdm

# get data
# Full cofea dataset
data_dir = join('data',"COCA",'COCA Text')
data_files = [filename for filename in glob.iglob(join(data_dir,'**/*.txt'), recursive=True)]
full_data = []
for file in data_files: # adjust index slice here for the desired source
    with open(file, encoding = 'utf-8') as f:
        full_data = full_data + f.read().splitlines()
        # if using json
        # data = json.load(f)
        # full_data.append(data)
        
save_name = 'COCA'
with open(join('data','phrases.txt'), encoding = 'utf-8') as f:
    phrases = f.read().splitlines()
out_dir = join('data','preprocessed')

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer
stop = stopwords.words('english')

def clean_doc(doc,p_check=False):
    # break documents up into sentences and make sure each token is separated by just one space
    doc = doc.strip()
    doc = re.sub('(\\n|\\t|\\s)+'," ",doc)
    if p_check:
        for p in phrases:
            psub = p.replace(' ','')
            doc = doc.replace(p,psub)
    sents = sent_tokenize(doc)
    sents = [ ' '.join([y.text for y in tokenizer(x)]).lower() for x in sents] 
    return sents

def run_and_save(fname):
    
    #train model
    model = gensim.models.Word2Vec( alpha=0.025, window=4,vector_size=300, min_count=10, workers=12, sg=1, hs=0, negative=5)
    model.build_vocab(gensim.models.word2vec.LineSentence(fname+'.txt'))
    model.train(gensim.models.word2vec.LineSentence(fname+'.txt'), total_examples=model.corpus_count, epochs=5)
    model.wv.save_word2vec_format(fname+ '.tmp')
    # save .wv.npy and .vocab
    vec = []
    w = codecs.open(fname + '.vocab', 'w', encoding='utf-8')
    vocab_size, embed_dim = None, None
    with codecs.open(fname + '.tmp', 'r', encoding='utf-8', errors='ignore') as r:
        for line in r:
            items = line.strip().split()
            if not vocab_size:
                assert(len(items) == 2)
                vocab_size, embed_dim = int(items[0]), int(items[1])
            else:
                assert(len(items) == embed_dim + 1)
                vec.append([float(item) for item in items[1:]])
                w.write('%s\n'%items[0])
    w.close()
    vec = np.array(vec, dtype=np.float)
    assert(vec.shape[0] == vocab_size)
    assert(vec.shape[1] == embed_dim)
    np.save(fname + '.wv.npy', vec)
    print('saved %s.wv.npy'%fname)
    print('saved %s.vocab'%fname)
    os.remove(fname + '.tmp')
    
out_name = save_name + '.txt' #change name based on source
# Save to text file so it can be efficiently used by word2vec
with open(join(out_dir,out_name), 'w',encoding = 'utf-8') as filehandle:
    for d in tqdm(full_data):
        # clean up the docs for processing
        sents = [clean_doc(d,True)]
        for s in sents:
            filehandle.write('%s\n' % s)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
out_name = join(out_dir,save_name) 
run_and_save(out_name)