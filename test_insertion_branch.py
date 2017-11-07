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

import sys


def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true


# loading our cohernce model
saved_model = sys.argv[1]
final_model = load_model(saved_model)


#geting parametes 
params = saved_model.split("/")[-1]
params = params.split("_")

w_size = int(params[4])
maxlen= int(params[3])
emb_size = int(params[2])

flist = params[9]
if flist == "None":
    fn = []
else:
    fn = [0,3,4]    #fn = range(0,10) #using feature

print("---------------------------------------------------------------------------")    
print("Load model: " + saved_model)
print("Parameters: w_size=" + str(w_size) + " maxlen=" + str(maxlen) + " emb_size=" +str(emb_size) + " features: " + flist)

#print('Loading vocab of the whole dataset...')
vocabs, E = cnet_helper.init_vocab(emb_size)


#find the maximum coherence score when inserting the sentence at position k
def insert(filename="", k = 0, w_size=3, maxlen=14000, vocabs=None, feats=None):
    lines = [line.rstrip('\n') for line in open(filename+ ".EGrid")]
    doc_size = cnet_helper.find_len(sent=lines[1])
    #print(doc_size)

    X_1 =  cnet_helper.load_branch_POS_EGrid(filename=filename, w_size=w_size, maxlen=maxlen , vocabs=vocabs, feats=fn )
    
    #the lowest coherence score of a document
    bestScore = -999999.999999
    bestPos = []

    perm = []
    perm.append(k)
    for i in range(0, doc_size):
        if i!=k:
            perm.append(i)

    for pos in range(0,doc_size):
        #compute coherence score for permuated         
        X_0 =  cnet_helper.load_branch_NEG_EGrid(filename=filename, w_size=w_size , maxlen=maxlen , vocabs=vocabs, feats=fn, perm=perm)
        #print(perm)
        y_pred = final_model.predict([X_1, X_0])
        n = len(y_pred)
        
        score_pos = 0.0
        score_neg = 0.0
        for i in range(0,n):
            score_pos += y_pred[i][0]
            score_neg += y_pred[i][1]
 
        print(" - At position " + str(pos) + " |--> pos vs. neg score: " + str("%0.4f" % score_pos) + " vs. " + str("%0.4f" % score_neg) )

        #if score_neg >= score_pos: # bad insertion, we want score_1 is always greater than score_0

        if(score_neg > bestScore):

            bestScore = score_neg
            bestPos = []
            bestPos.append(pos)
        elif score_neg == bestScore:
            bestPos.append(pos)                
        
        if pos < doc_size-1:
            perm[pos] = perm[pos+1]
            perm[pos+1] = k
        #print(bestScore)

    return bestPos


totalPerf = 0
totalIns = 0 
docAvgPerf = 0.0

#main function here
list_of_files = [line.rstrip('\n') for line in open("final_data/CNET/x_cnet.4test")]
totalPerf = 0
for file in list_of_files:
    # process each test document
    doc_size = cnet_helper.find_doc_size(file+".EGrid");
    print("---------------------------------------------------------------------------")    
    print(str(file))    
    
    perfects = 0;
    for k in range(0, doc_size):
        print ("Insert sent " + str(k) + "...")
        bestPos = insert(file, k, w_size=w_size, maxlen=maxlen,vocabs=vocabs, feats=fn)

        print ("==> Having best coherrent positions: " + str(bestPos))
        if k in bestPos:
            perfects = perfects + 1

    totalPerf = totalPerf + perfects
    totalIns = totalIns + doc_size
    docAvgPerf = docAvgPerf + perfects / doc_size;
    print ("Document perfect: " + str(perfects) + " of " + str(doc_size))

print ("\nSummary...")  
print (" -Perfect: " + str(totalPerf)) 
print (" -Perfect by line: " + str(totalPerf/totalIns))    
print (" -Perfect by doc: " + str(docAvgPerf/len(list_of_files)))    


