from keras.layers import Flatten, Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
from keras.models import Model
from keras import objectives
from keras.preprocessing import sequence

import numpy as np
import data_helper
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
w_size = 8
maxlen=4000

#hyper paramere for cnn
nb_filter = 150
filter_length = w_size
pool_length = 7
dropout_ratio = 0.5
hidden_size = 250
emb_size = 100

opt='adam'

#loading entity-gird for pos and neg documents
X_train_1, X_train_0, max_ent_num_train, max_sent_num_train	= data_helper.load_and_numberize_Egrid(filelist="list.train", 
            perm_num = p_num, maxlen=maxlen, window_size=w_size, ignore=0)
X_dev_1, X_dev_0, max_ent_num_dev, max_sent_num_dev	= data_helper.load_and_numberize_Egrid(filelist="list.test", 
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

print("---------------------------------------------------------")	
print("Loading grid data done...")
print("Num of documents: ")
print("Num of traing pairs: " + str(num_train))
print("Num of dev pairs: " + str(num_dev))
print("Num of permutation: 20") 
print("The maximum in length for CNN: " + str(maxlen))
print("The maximum num of entities: " + str(max_ent_num_train))
print("The maximum num of sentence in a doc: " + str(max_sent_num_train))

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
np.random.seed(133)
np.random.shuffle(X_train_1)
np.random.seed(133)
np.random.shuffle(X_train_0)

#loading embeddings
E = data_helper.load_embeddings(emb_size=emb_size)

# first, define a CNN model for sequence of entities 
# input of sequences of X,O,S,-,P between 1 and 5
sent_input = Input(shape=(maxlen,), dtype='int32', name='sent_input')

# embedding layer encodes the input into sequences of 300-dimenstional vectors. 
x = Embedding(output_dim=emb_size, weights=[E], input_dim=5, input_length=maxlen)(sent_input)

# add a convolutiaon 1D layer
x = Convolution1D(nb_filter=nb_filter, filter_length = filter_length, border_mode='valid', 
            activation='relu', subsample_length=1)(x)

# add max pooling layers
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
final_model.fit([X_train_1, X_train_0], y_train_1, validation_data=([X_dev_1, X_dev_0], y_dev_1), nb_epoch=25,
 					callbacks=[histories],verbose=1, batch_size=32)

print(histories.losses)
print(histories.accs)










