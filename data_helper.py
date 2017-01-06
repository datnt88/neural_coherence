from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter

def load_and_numberize_Egrid(filelist="list_of_grid.txt", perm_num = 3, maxlen=None):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    for file in list_of_files:
        print(file)
        lines = [line.rstrip('\n') for line in open(file)]

        tmp_str = "0 0 0 0 "
        for line in lines:
            # remove entity name merge them in to a single string
            tmp_str = tmp_str + line[21:] + " 0 0 0 0 0 "
        #print(tmp_str)
        for i in range (0, perm_num): #stupid code
            sentences_1.append(tmp_str)

    # process permuted entity grid
    sentences_0 = []
    for file in list_of_files:
        for i in range(1,perm_num+1):
            lines = [line.rstrip('\n') for line in open(file+".perm-"+str(i)+".txt")]    
            tmp_str = "0 0 0 0 "
            for line in lines:
                # remove entity name merge them in to a single string
                tmp_str = tmp_str + line[21:] + " 0 0 0 0 0 "
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

    # maybe we have to load a fixed embeddeings for each S,O,X,- the representation of 0 is zeros vector
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (5, 300) )
    E[0] = 0
    
    return X_1, X_0, E

def load_and_numberize_data(path="../data/", maxlen=None, seed=113, init_type="random", dev_train_merge=0):
	# loading the entity-grid data
    vocab_list = ['0','S','O','X','-']

    sentences_train = []
    
    sentences_test  = []

    sentences_dev   = []

    for filename in glob.glob(os.path.join(path, '*.csv')):
#       print "Reading vocabulary from" + filename
        reader  = csv.reader(open(filename, 'rb'))

        for rowid, row in enumerate (reader):
            #if rowid == 0: #header
            #    continue
            if re.search("train.csv", filename.lower()):    
                sentences_train.append(row[0])    

            elif re.search("test.csv", filename.lower()):    
                sentences_test.append(row[0])    

            elif re.search("dev.csv", filename.lower()):    
                sentences_dev.append(row[0]) 
    
    print "Nb of docs: train: " + str (len(sentences_train)) + " test: " + str (len(sentences_test)) + " dev: " + str (len(sentences_dev))
    print "Total vocabulary size: " + str (len(vocab_list)) 

    #Create vocab dictionary that maps word to ID
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

     # Numberize the sentences
    X_train = numberize_sentences(sentences_train, vocab_idmap)
    X_test  = numberize_sentences(sentences_test,  vocab_idmap)
    X_dev   = numberize_sentences(sentences_dev,   vocab_idmap)

    #assert len(X_train) == len(y_train) or len(X_test) == len(y_test) or len(X_dev) == len(y_dev)

    # we do not need to adjust the index since we already add padding in very first begining
    # but we still have to set the maxlen, remove sentences which length < MaxLen
    X_train = adjust_index(X_train, maxlen=maxlen)
    X_test  = adjust_index(X_test,  maxlen=maxlen)
    X_dev   = adjust_index(X_dev,   maxlen=maxlen)

    if dev_train_merge:
        X_train.extend(X_dev)
        #y_train.extend(y_dev)
    
    # maybe we have to load a fixed embeddeings for each S,O,X,- the representation of 0 is zeros vector
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (5, 300) )
    E[0] = 0
    
    return X_train, X_test, X_dev, E


def numberize_sentences(sentences, vocab_idmap):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
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





