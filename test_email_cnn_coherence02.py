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

print('Loading vocab of the whole dataset...')
vocab = email_helper.init_vocab()
print vocab
print "--------------------------------------------------------"

X_test_1, X_test_0, E , test_f_tracks   = email_helper.load_testing_data("dataset/CNET/new_m_cnet.test",
            perm_num = 20, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=fn)
print("loading test data done...")

num_test = len(X_test_1)
y_test_1 = [1] * num_test


#print(final_model.summary())
print("---------------------------------------------------------")	
print("Testing model...")

saved_model = sys.argv[1]
final_model = load_model(saved_model)
y_pred = final_model.predict([X_test_1, X_test_0])

y_pred = final_model.predict([X_test_1, X_test_0])
ties = 0
wins = 0

for docId in range(0, max(test_f_tracks) + 1):
	indexes = [i for i,idx in enumerate(test_f_tracks) if idx == docId]

        pos_score = 0.0
        neg_score = 0.0
        for i in indexes:
		pos_score += y_pred[i][0]
                neg_score += y_pred[i][1]

        if pos_score > neg_score:
                wins = wins + 1
        elif pos_score == neg_score:
                ties = ties + 1

# number of test set
n = max(test_f_tracks) + 1

print("Perform on test set ...!")
print(" -Wins: " + str(wins) + " Ties: "  + str(ties))
loss = n - (wins+ties)
#recall = wins/n;
prec = wins/(wins + loss)
#f1 = 2*prec*recall/(prec+recall)

print(" -Test acc: " + str(wins/n))


