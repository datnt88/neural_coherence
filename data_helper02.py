from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter

def remove_entity(sent=""):
    x = sent.split()
    count = x.count('X') + x.count('S') + x.count('O') #counting the number of entities
    if count <3: #remove lesss ferequent entities
        return ""
    x = x[1:]
    return ' '.join(x)

def get_ETransition(sent=""):
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



def load_and_numberize_Egrid(filelist="list_of_grid.txt", perm_num = 3, maxlen=None, window_size=3, ignore=0):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []

    e_count_list = []

    for file in list_of_files:
        #print(file)
        lines = [line.rstrip('\n') for line in open(file)]

        grid_1 = "0 "* window_size
        e_count = 0
        for line in lines:
            # merge the grid of positive document 
            e_trans = get_ETransition(sent=line)
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
                e_count = e_count + 1
        	
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+"-"+str(i))]    
            grid_0 = "0 "* window_size
            for p_line in permuted_lines:
                e_trans_0 = get_ETransition(sent=p_line)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

            if grid_0 != grid_1:
                p_count = p_count + 1
                sentences_0.append(grid_0)
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)
            
       #update new number of entity
        e_count_list.append(e_count)

    assert len(sentences_0) == len(sentences_1)
#    with open('e_count_list_after.txt','w') as f:
#        f.write('\n'.join([str(n) for n in e_count_list])+'\n')

    # numberize_data
    vocab_list = ['0','S','O','X','-']
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

     # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, ignore=ignore, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, ignore=ignore, window_size=window_size)

    return X_1, X_0


def load_embeddings(emb_size=300):
    # maybe we have to load a fixed embeddeings for each S,O,X,- the representation of 0 is zeros vector
    np.random.seed(2016)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (5, emb_size))
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

def adjust_index(X, maxlen=None,ignore=0, window_size=3):

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        for x in X:

            if len(x) <= maxlen:
                new_X.append(x)
            elif ignore == 0:
            	tmp = x[0:maxlen]
            	tmp[maxlen-window_size:maxlen] = ['0'] * window_size
            	new_X.append(tmp)

        X = new_X

    return X





