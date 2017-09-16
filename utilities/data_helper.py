from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter
import itertools
from keras.preprocessing import sequence


#initilize basic vocabulary for cnn, this will change when using features
def init_vocab(emb_size):
    vocabs =['0','S','O','X','-']

    #v2s = list(itertools.product('SOX-', repeat=2))
    #for tupl in v2s:
    #    vocabs.append(''.join(tupl))

    #v3s = list(itertools.product('SOX-', repeat=3))
    #for tupl in v3s:
    #    vocabs.append(''.join(tupl))

    #v4s = list(itertools.product('SOX-', repeat=4))
    #for tupl in v4s:
    #    vocabs.append(''.join(tupl))

    np.random.seed(2017)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocabs), emb_size))
    E[0] = 0

    return vocabs, E


#loading the grid with normal CNN
def load_and_numberize_egrids(filelist="list_of_grid_pair.txt", maxlen=None, w_size=3, vocabs=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_pairs = [line.rstrip('\n') for line in open(filelist)]
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []

    
    for pair in list_of_pairs:
        #print(pair)

        pos_doc = pair.split("\t")[0]
        neg_doc = pair.split("\t")[1]

        #loading Egrid for POS document
        grid_1 = load_egrid(pos_doc,w_size)
        grid_0 = load_egrid(neg_doc,w_size)
        

        if grid_0 != grid_1:
                sentences_1.append(grid_1)
                sentences_0.append(grid_0)
                  
    #assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)    

    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    return X_1, X_0

##loading Egrid for a document
def load_egrid(filename,w_size):

    lines = [line.rstrip('\n') for line in open(filename)]
    grid = "0 "* w_size
    for line in lines:
        # merge the grid of positive document 
        e_trans = get_eTrans(sent=line)
        if len(e_trans) !=0:
            grid = grid + e_trans + " " + "0 "* w_size

    return grid


# get each transition for each entity (each line in egrid file)
def get_eTrans(sent=""):
    x = sent.split()
    x = x[1:]
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    return ' '.join(x)


def numberize_sentences(sentences, vocab_idmap):  
    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        #print(sid)
        for wrd in sent.split():
            if wrd in vocab_idmap:
                wrd_id = vocab_idmap[wrd]  
            else:
                wrd_id = 0
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id  

def adjust_index(X, maxlen=None, window_size=3):
    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        for x in X:

            if len(x) > maxlen:
                #print("************* Maxlen of whole dataset: " + str(len(x)) )
                tmp = x[0:maxlen]
                tmp[maxlen-window_size:maxlen] = ['0'] * window_size
                new_X.append(tmp)
            else:
                new_X.append(x)

        X = new_X

    return X



 




