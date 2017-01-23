from __future__ import division
from keras.layers import AveragePooling1D, Flatten, Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
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
maxlen=15000

#hyper paramere for cnn standard is: 150, 6, 250, 100
nb_filter = 150
filter_length = w_size
pool_length = 6
dropout_ratio = 0.5
hidden_size = 250
emb_size = 100 #100, best performacen, 

opt='rmsprop'


#loading vocab, enity and embedding
fn = [2,3,4] #using feature
vocab = data_helper02.load_all(filelist="final_data/list.all.docs",fn=fn)


#loading entity-gird for pos and neg documents
X_train_1, X_train_0, E = data_helper02.load_and_numberize_Egrid_with_Feats(filelist="final_data/list.train.docs", 
            perm_num = p_num, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=fn)
X_dev_1, X_dev_0, E    = data_helper02.load_and_numberize_Egrid_with_Feats(filelist="final_data/list.dev.docs", 
            perm_num = p_num, maxlen=maxlen, window_size=w_size, E=E ,vocab_list=vocab, emb_size=emb_size, fn=fn)
X_test_1, X_test_0, E    = data_helper02.load_and_numberize_Egrid_with_Feats(filelist="final_data/list.test.docs.final", 
            perm_num = p_num, maxlen=maxlen, window_size=w_size, E=E ,vocab_list=vocab, emb_size=emb_size, fn=fn)

num_train = len(X_train_1)
num_dev   = len(X_dev_1)
num_test  = len(X_test_1)
#assign Y value
y_train_1 = [1] * num_train 
y_dev_1 = [1] * num_dev 
y_test_1 = [1] * num_test 

print("---------------------------------------------------------")	
print("Loading grid + features data done...")
print("Num of traing pairs: " + str(num_train))
print("Num of dev pairs: " + str(num_dev))
print("Num of test pairs: " + str(num_test))
print("Num of permutation in train: " + str(p_num)) 
print("The maximum in length for CNN: " + str(maxlen))


# the output is always 1??????
y_train_1 = np_utils.to_categorical(y_train_1, 2)
y_dev_1 = np_utils.to_categorical(y_dev_1, 2)
y_test_1 = np_utils.to_categorical(y_test_1, 2)

#randomly shuffle the training data
np.random.seed(113)
np.random.shuffle(X_train_1)
np.random.seed(113)
np.random.shuffle(X_train_0)


# first, define a CNN model for sequence of entities 
sent_input = Input(shape=(maxlen,), dtype='int32', name='sent_input')

# embedding layer encodes the input into sequences of 300-dimenstional vectors. 
x = Embedding(output_dim=emb_size, weights=[E], input_dim=len(vocab), input_length=maxlen)(sent_input)

# add a convolutiaon 1D layer
#x = Dropout(dropout_ratio)(x)
x = Convolution1D(nb_filter=nb_filter, filter_length = filter_length, border_mode='valid', 
            activation='relu', subsample_length=1)(x)

# add max pooling layers
#x = AveragePooling1D(pool_length=pool_length)(x)
x = MaxPooling1D(pool_length=pool_length)(x)
x = Dropout(dropout_ratio)(x)
x = Flatten()(x)
#x = Dense(hidden_size, activation='relu')(x)
x = Dropout(dropout_ratio)(x)

# add latent coherence score
out_x = Dense(1, activation='linear')(x)
shared_cnn = Model(sent_input, out_x)

# Inputs of pos and neg document
pos_input = Input(shape=(maxlen,), dtype='int32', name="pos_input")
neg_input = Input(shape=(maxlen,), dtype='int32', name="neg_input")

# these two models will share eveything from shared_cnn
pos_branch = shared_cnn(pos_input)
neg_branch = shared_cnn(neg_input)

concatenated = merge([pos_branch, neg_branch], mode='concat',name="coherence_out")
# output is two latent coherence score

final_model = Model([pos_input, neg_input], concatenated)

#final_model.compile(loss='ranking_loss', optimizer='adam')
final_model.compile(loss={'coherence_out': ranking_loss}, optimizer=opt)

# setting callback
histories = my_callbacks.Histories()

print(shared_cnn.summary())
print(final_model.summary())

print("---------------------------------------------------------")	
print("Training model...")

for i in range(1,50):
    saved_model = "./ext_cnn_saved_models/ext-CNN-maxlen" + str(maxlen) + "-w_size" + str(w_size) + "_MaxPool" + str(pool_length) + "-epoch-" + str(i) +".h5"
    final_model.fit([X_train_1, X_train_0], y_train_1, validation_data=([X_dev_1, X_dev_0], y_dev_1), nb_epoch=1,
 					verbose=1, batch_size=32, callbacks=[histories])
    final_model.save(saved_model)

    y_pred = final_model.predict([X_test_1, X_test_0])
        
    ties = 0
    wins = 0
    n = len(y_pred)
    for i in range(0,n):
        if y_pred[i][0] > y_pred[i][1]:
            wins = wins + 1
        elif y_pred[i][0] == y_pred[i][1]:
            ties = ties + 1
    print("Perform on test set after Epoch: " + str(i))    
    print(" -Wins: " + str(wins) + " Ties: "  + str(ties))
    loss = n - (wins+ties)
    recall = wins/n;
    prec = wins/(wins + loss)
    f1 = 2*prec*recall/(prec+recall)

    print(" -Test acc: " + str(wins/n))
    print(" -Test f1 : " + str(f1))


print("Loss information:...")
print(histories.losses)
print(histories.accs)










