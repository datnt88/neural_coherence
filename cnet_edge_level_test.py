from __future__ import division

from keras.models import load_model
from keras.layers import Dense, Flatten, Input, Embedding, MaxPooling1D, Dropout, Conv1D
from keras.layers.merge import concatenate
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
    #ranking_loss without tree distance
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true


def ranking_loss_with_penalty(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    dist = y_pred[:,2]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(dist + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

def compute_best_tree(trained_model=None, file="", maxlen=1000, w_size=5, vocabs=[], emb_size=50):

    #get original tree 
    o_pairs, p_pairs, nPost = cnet_helper.get_pos_and_neg_pairs(file) 

    best_pairs = []

    for i in range(3,nPost+1): #recontruct the tree
        print "    --------------------------"
        print "    Reconstuct post: " , i
        pre_posts = range(1,i)
        
        max_score = -10E10
        best_prev = 1

        for pre in pre_posts:
            X, dist = cnet_helper.load_one_edge_only(file=file, pair=str(pre) + "." + str(i), maxlen=maxlen, 
                w_size=w_size, vocabs=vocabs, emb_size=emb_size)
            
            y_pred = trained_model.predict([X,X])
            score = y_pred[0][0]

            if score > max_score:
                max_score = score
                best_prev = pre

            print "\tscore of "+ str(pre) + "." + str(i) + ": ", my_format(score)

        print "    Found: " + str(best_prev) + "." + str(i)

        best_pairs.append(str(best_prev) + "." + str(i))

    best_pairs.append('1.2')
    best_pairs = sorted(best_pairs)
    
    true_prediction = [i for i in best_pairs if i in o_pairs]
    edge_acc = len(true_prediction)/len(o_pairs)
    tree_count = 0
    if edge_acc == 1:
        tree_count = 1 

    print "Org tree: ", o_pairs, " Best tree: ", best_pairs

    print "Tree prediction: ", tree_count, " Edge acc: ", my_format(edge_acc)
    return tree_count, edge_acc
    


#---------------------------------------------------------------------    
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

list_of_files = [line.rstrip('\n') for line in open("./final_data/CNET/x_cnet.test")]
count = 0



acc_edge = 0.0
count = 0 

for file in list_of_files:
    #procee each test
    print "=========================================================================="
    print "Processing: " + file
    
    #find the best tree base on coherence score
    tree_count, tree_edge_acc = compute_best_tree(trained_model=final_model, file=file, maxlen=maxlen, w_size=w_size, vocabs=vocabs, emb_size=emb_size)
    count += tree_count
    
    acc_edge += tree_edge_acc 

k = len(list_of_files)


print "-----------------------------------------------------------------"
print "Right tree prediction: " ,str(count)
print "Acc Tree: " , count/k
print "Acc edge: " , acc_edge/k 
























