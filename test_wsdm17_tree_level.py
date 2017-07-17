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

from utilities import cnet_helper
from utilities import my_callbacks
from utilities import gen_trees
from utilities import baselines

import sys

def my_format(x):
    return str("{0:.4f}".format(x))

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

def compute_score(trained_model=None, file="", tree=[], maxlen=1000, w_size=5, vocab=[], emb_size=50):
    cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
    cmtIDs = [int(i) for i in cmtIDs] 
    
    branches = cnet_helper.get_tree_struct(cmtIDs=cmtIDs,tree=tree) # get branches with sentID

    level_dict = {}
    for branch in branches:
        for i,j in enumerate(branch):
            level_dict[j] = i
    sentDepths = level_dict.values()
    #print sentDepths

    X, dist = cnet_helper.load_one_tree_only(file=file, sent_levels=sentDepths, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=None)
    y_pred = trained_model.predict([X, X, dist])
    
    return y_pred[0][0]
    
#load the best model
#saved_final_N01/CNN.20_0.5_100_10000_5_150_6_32_FNone_ep.5.h5
model_path = sys.argv[1]
final_model = load_model(model_path)
print "Running standard thread reconstruction...!"
print "Loading model: " + model_path


#geting parametes 
params = model_path.split("/")[-1].split("_")
print "Prameters:", params

w_size = int(params[5])
maxlen= int(params[4])
emb_size = int(params[3])
fn = []    #fn = range(0,10) #using feature


print('Loading vocab of the whole dataset...')
vocabs, E = cnet_helper.init_vocab(emb_size)

list_of_files = [line.rstrip('\n') for line in open("./final_data/CNET/p5_s_cnet.test_tmp")]
count = 0


f1_edge = 0.0
acc_edge = 0.0
x_count = 0 

for file in list_of_files:
    #procee each test
    print "=========================================================================="
    print "Processing: " + file
    
    #get original tree
    x_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
    org_tree = []
    cmtIDs02 = []

    for i in x_tree:
        org_tree += [''.join(i.split("."))]
        cmtIDs02 += [int(j) for j in i.split(".")]

    gold_score = compute_score(trained_model=final_model, file=file, tree=org_tree, maxlen=maxlen, w_size=w_size, vocab=vocabs, emb_size=emb_size)

    #working which candidate trees
    # sepcial for tree with have more than 4 coments
    nCmt = max(cmtIDs02)
    max_score = -10E10
    best_tree = []
    
    p_trees = gen_trees.gen_tree_branches(nCmt) # no pruning
    #p_trees = gen_trees.prune_trees(n=nCmt)
    #p_trees = gen_trees.get_top_possible_trees(file=file)

    for p_tree in p_trees:
        #print p_tree
        p_score = compute_score(trained_model=final_model, file=file, tree=p_tree, maxlen=maxlen, w_size=w_size, vocab=vocabs, emb_size=emb_size)

        print "\tp-score: " + my_format(p_score)  + " \tTree: " + str(p_tree)
        if p_score > max_score:
            max_score = p_score
            best_tree = p_tree

    #computing accuracy
    check = "LOSE"
    if gold_score == max_score:
        count +=1
        check = "WIN"
    
    print file.split("/")[3] + " Gold: " + my_format(gold_score) +  " Best: "  + my_format(max_score) + "\t" + check  + "\t " +  str(org_tree)+ "\t" +str(best_tree) 


    #computing acc, F1 
    org_tree = ['.'.join(i) for i in org_tree]
    best_tree = ['.'.join(i) for i in best_tree]

    org_n, org_edge =  baselines.getOriginalInfo(tree=org_tree)
    p_n, p_edge =  baselines.getOriginalInfo(tree=best_tree)

    if '-'.join(org_tree) == '-'.join(best_tree):
        x_count = 0

    assert (org_n == p_n)
    f1_edge += baselines.f1_score(org_edge,p_edge, average='micro') 
    acc_edge += baselines.compute_edge_acc(org_edge,p_edge)

k = len(list_of_files)

print "--------------------------------------------"
print "Acc Tree: " + str(count)
print "X count : " + str(x_count)
print "Acc edge: " , acc_edge/k 
print "F1 edge : " , f1_edge/k























