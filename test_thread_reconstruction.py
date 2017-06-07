from __future__ import division
from keras.models import load_model
from keras.layers import Flatten, Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import objectives
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

import numpy as np
from keras.utils import np_utils
from keras import backend as K

from utilities import email_helper
from utilities import my_callbacks
from utilities import gen_trees


import sys

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

#parameter for data_helper
p_num = 20
w_size = 5
maxlen=14000
emb_size = 50
fn = [0,3,4]    #fn = range(0,10) #using feature

#load the best model
final_model = load_model("./saved_models/CNN.20_0.5_50_14000_5_150_6_64_FNone_ep.2.h5")

print('Loading vocab of the whole dataset...')
vocab = email_helper.init_vocab()
#print vocab
print "--------------------------------------------------------"

list_of_files = [line.rstrip('\n') for line in open("./dataset/CNET/list.test_reconstruction")]
count = 0

for file in list_of_files:
    #procee each test
    print file
    X_org = email_helper.load_original_tree(file=file, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=fn)
    #compute the coherence score cor the original tree
    y_pred = final_model.predict([X_org, X_org])
    gold_score = 0.0
    for i in range(len(X_org)):
        gold_score += y_pred[i][0]
        
    #gold_score = gold_score/len(X_org)
    print "Gold score: " + str(gold_score)
    

    #processing each possible tree candidate
    cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
    cmtIDs = [int(i) for i in cmtIDs] 
    p_trees =  gen_trees.gen_tree_branches(max(cmtIDs))
    
    max_score = 0.0
    for p_tree in p_trees:

        x_tree = [] # tree with sentence ID
        #print cmtIDs
        for branch in p_tree:
            
            #then compare to the original one
            #print list(branch)
            sentIDs = []
            for cmtID in list(branch):   
                sentIDs += [ i for i, id_ in enumerate(cmtIDs) if id_ == int(cmtID)]
            x_tree.append(sentIDs)

        #print x_tree
            #
        X_perm = email_helper.load_permuted_tree(file=file, tree=x_tree, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=fn)

        #compute the cohenrence score for the permutation tree
        y_pred = final_model.predict([X_perm, X_perm])
        p_score = 0.0
        for i in range(len(X_perm)):
            p_score += y_pred[i][0]
        
        #p_score = p_score/len(X_perm)
        if p_score > max_score:
            max_score = p_score

        print p_score

    if gold_score >= max_score:
        count +=1

    print str(gold_score)
    print str(max_score)
    print "-------------------------------------"

print count

    #predict score for the original and permitation threads
    #the original score should winn over any permutation one






















