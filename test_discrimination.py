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

from utilities import data_helper
from utilities import my_callbacks


import sys

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

#parameter for data_helper
p_num = 20
w_size = 6
maxlen=14000
emb_size = 100
fn = []    #fn = range(0,10) #using feature

    
print('Loading vocab of the whole dataset...')
vocab = data_helper.load_all(filelist= "final_data/list.all.0001.docs",fn=fn)
print(vocab)

X_test_1, X_test_0, E = data_helper.load_and_numberize_Egrid_with_Feats(filelist="final_data/list.test.docs.final", 
            perm_num = p_num, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=fn)

num_test = len(X_test_1)
y_test_1 = [1] * num_test


#print(final_model.summary())
print("---------------------------------------------------------")	
print("Testing model...")

saved_model = sys.argv[1]
final_model = load_model(saved_model)
y_pred = final_model.predict([X_test_1, X_test_0])
    
ties = 0
wins = 0
n = len(y_pred)
#print (n)
#print (y_pred.shape)

for i in range(0,n):
    if y_pred[i][0] > y_pred[i][1]:
        wins = wins + 1
    elif y_pred[i][0] == y_pred[i][1]:
        ties = ties + 1
    
print("\n -Wins: " + str(wins) + " Ties: "  + str(ties))
loss = n - (wins+ties)
recall = wins/n;
prec = wins/(wins + loss)
f1 = 2*prec*recall/(prec+recall)

print(" - Test acc: " + str(wins/n))
print(" - Test f1 : " + str(f1))



