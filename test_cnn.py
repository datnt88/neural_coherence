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


def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    #loss = -K.sigmoid(pos-neg) # use 
    loss = K.maximum(1.0 + neg - pos, 0.0) #if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true

#parameter for data_helper
p_num = 20
w_size = 6
maxlen=10676

#hyper paramere for cnn
nb_filter = 150
filter_length = w_size
pool_length = 6
dropout_ratio = 0.5
hidden_size = 250
emb_size = 100

opt='rmsprop'

#loading entity-gird for pos and neg documents
X_train_1, X_train_0 = data_helper02.load_and_numberize_Egrid(filelist="final_data/list.test", 
            perm_num = p_num, maxlen=maxlen, window_size=w_size, ignore=0)
X_dev_1, X_dev_0 = data_helper02.load_and_numberize_Egrid(filelist="final_data/list.dev", 
            perm_num = 20, maxlen=maxlen, window_size=w_size, ignore=0)
#X_test_1, X_test_0	= data_helper.load_and_numberize_Egrid(filelist="list_of_test.txt", perm_num = 3)


num_train = len(X_train_1)
num_dev   = len(X_dev_1)
#num_test  = len(X_test_1)

#assign Y value
y_train_1 = [1] * num_train 
y_dev_1 = [1] * num_dev 
#y_test_1 = [1] * num_test 

# find the maximum length for padding
maxlen = max(len(l) for l in X_train_1)
print(maxlen)
maxlen_dev = max(len(l) for l in X_dev_1)
print(maxlen_dev)
if maxlen_dev > maxlen:
	maxlen = maxlen_dev 
maxlen = 10676

print("---------------------------------------------------------")	
print("Loading grid data done...")
print("Num of documents: ")
print("Num of traing pairs: " + str(num_train))
print("Num of dev pairs: " + str(num_dev))
print("Num of permutation: 20") 
print("The maximum in length for CNN: " + str(maxlen))
#print("The maximum num of entities: " + str(max_ent_num_train))
#print("The maximum num of sentence in a doc: " + str(max_sent_num_train))

# let say default is 500
#maxlen=500

X_train_1 = sequence.pad_sequences(X_train_1, maxlen)
X_dev_1   = sequence.pad_sequences(X_dev_1, maxlen)
#X_test_1  = sequence.pad_sequences(X_test_1, maxlen)

X_train_0 = sequence.pad_sequences(X_train_0, maxlen)
X_dev_0   = sequence.pad_sequences(X_dev_0, maxlen)
#X_test_0  = sequence.pad_sequences(X_test_0, maxlen)


# the output is always 1??????
y_train_1 = np_utils.to_categorical(y_train_1, 2)
y_dev_1 = np_utils.to_categorical(y_dev_1, 2)
#y_train_1 = np.ones(num_train)
#y_dev_1  = np.ones(num_dev)

#randomly shuffle the training data
#np.random.seed(133)
#np.random.shuffle(X_train_1)
#np.random.seed(133)
#np.random.shuffle(X_train_0)



#loading model
#final_model = load_model('my_model.h5')

#print(final_model.summary())
print("---------------------------------------------------------")	
print("Testing model...")

for ep in range(1,25):
#    checkpointer = ModelCheckpoint(filepath="./tmp/weights-" + str(i) +".hdf5", verbose=1, save_best_only=True)
#    final_model.fit([X_train_1, X_train_0], y_train_1, validation_data=([X_dev_1, X_dev_0], y_dev_1), nb_epoch=1,
# 					verbose=1, batch_size=32, callbacks=[histories,checkpointer])
    saved_model = "./saved_models/cnn-egrid-epoch-" + str(ep) +".h5"
    final_model = load_model(saved_model)
    y_pred = final_model.predict([X_train_1, X_train_0])
    
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



