from __future__ import division
from keras.models import load_model
from keras.layers import Flatten, Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import objectives
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

import numpy as np
import data_helper02
from keras.utils import np_utils
from keras import backend as K

import my_callbacks
import sys


def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

w_size = 6
maxlen=10676
# loading our cohernce model
saved_model = "./saved_models/cnn-egrid-epoch-4.h5"
final_model = load_model(saved_model)


#find the maximum coherence score when inserting the sentence at position k
def insert(filename="",k = 0,w_size=3,maxlen=maxlen):
    lines = [line.rstrip('\n') for line in open(filename)]
    doc_size = data_helper02.find_len(sent=lines[1])

    X_1 =  data_helper02.load_POS_EGrid(filename=filename,w_size=w_size)
    X_1 = sequence.pad_sequences(X_1, maxlen)
    
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
        X_0 =  data_helper02.load_NEG_EGrid(filename=filename, perm=perm, w_size=w_size)
        X_0 = sequence.pad_sequences(X_0, maxlen)
        
        y_pred = final_model.predict([X_1, X_0])

        score_pos = y_pred[0][0]
        score_neg= y_pred[0][1]
 
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
list_of_files = [line.rstrip('\n') for line in open(sys.argv[1])]
totalPerf = 0
for file in list_of_files:
    # process each test document
    doc_size = data_helper02.find_doc_size(file);
    print("------------------------------------------------------------")    
    print(str(file))    
    
    perfects = 0;
    for k in range(0, doc_size):
        print ("Insert sent " + str(k) + "...")
        bestPos = insert(file, k, w_size=w_size, maxlen=maxlen)
        print ("==> Having best coherrent positions: " + str(bestPos))
        if k in bestPos:
            perfects = perfects + 1

    totalPerf += totalPerf + perfects
    totalIns += totalIns + doc_size
    docAvgPerf += docAvgPerf + perfects / doc_size;
    print ("Document perfect: " + str(perfects) + " of " + str(doc_size))

print ("\nSummary...")  
print (" -Perfect: " + str(totalPerf)) 
print (" -Perfect by line: " + str(totalPerf/totalIns))    
print (" -Perfect by doc: " + str(docAvgPerf/len(list_of_files)))    


