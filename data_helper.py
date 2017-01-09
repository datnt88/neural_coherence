from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter

def load_and_numberize_Egrid(filelist="list_of_grid.txt", perm_num = 3, maxlen=None, window_size=3):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    max_entity_num = 0
    max_sent_num = 0
    for file in list_of_files:
        #print(file)
        lines = [line.rstrip('\n') for line in open(file)]

        if  len(lines) > max_entity_num:  # finding the max number of entity for the whole collection
            max_entity_num = len(lines)
        tmp_sent = lines[1][21:]
        sent_num = (len(tmp_sent) + 1)/2
        if sent_num  > max_sent_num:
            max_sent_num = sent_num

        tmp_str = "0 "* window_size
        for line in lines:
            # remove entity name merge them in to a single string
            tmp_str = tmp_str + line[21:] + " " + "0 "* window_size
        #print(tmp_str)
        for i in range (0, perm_num): #stupid code
            sentences_1.append(tmp_str)

    # process permuted entity grid
    sentences_0 = []
    for file in list_of_files:
        for i in range(1,perm_num+1):
            lines = [line.rstrip('\n') for line in open(file+"-"+str(i))]    
            tmp_str = "0 "* window_size
            for line in lines:
                # remove entity name merge them in to a single string
                tmp_str = tmp_str + line[21:] + " " + "0 "* window_size
            sentences_0.append(tmp_str)

    #print(len(sentences_1))
    #print(len(sentences_0))
    #print(sentences_0)
    assert len(sentences_0) == len(sentences_1)

    # numberize_data
    vocab_list = ['0','S','O','X','-']
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

     # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen)
    X_0  = adjust_index(X_1,  maxlen=maxlen)

    return X_1, X_0, max_entity_num, max_sent_num


def load_embeddings():
    # maybe we have to load a fixed embeddeings for each S,O,X,- the representation of 0 is zeros vector
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (5, 300))
    E[0] = 0
    return E   
 

def numberize_sentences(sentences, vocab_idmap):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        #print(sid)
        for wrd in sent.split():
            wrd_id = vocab_idmap[wrd]  
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id  

def adjust_index(X, maxlen=None):

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)

        X      = new_X

    return X





