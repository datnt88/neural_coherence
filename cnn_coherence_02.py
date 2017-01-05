from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution1D, MaxPooling1D, Dropout
from keras.models import Model

import numpy as np
import utilities 

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = -K.sigmoid(pos-neg) # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to use margin ranking loss
    return K.mean(loss) + 0 * y_true




#loading pos entity-gird 
X_train_1, X_test_1, X_dev_1, E = utilities.load_and_numberize_data(path="./data/1/", dev_train_merge=0)

#loading permuted entity-gird 
X_train_0, X_test_0, X_dev_0, E = utilities.load_and_numberize_data(path="./data/0/", dev_train_merge=0)

assert len(X_train_1) == len(X_train_0) or len(X_test_1) == len(X_test_0) or len(X_dev_1) == len(y_dev_0)

num_train = len(X_train_1)
num_dev = len(X_dev_1)
num_test = len(X_test_1)

#assign Y value
y_train_1 = [1] * num_train 
y_train_0 = [0] * num_train 

y_dev_1 = [1] * num_dev 
y_dev_0 = [0] * num_dev 

y_test_1 = [1] * num_test 
y_test_0 = [0] * num_test

#randomly shuffle the training data
np.random.seed(133)
np.random.shuffle(X_train_1)
np.random.seed(133)
np.random.shuffle(X_train_0)

print(y_train_0)


#hyper parameres
nb_filter = 150
filter_length = 3
pool_length = 4
dropout_ratio = 0.5
hidden_size = 250

# first, define a CNN model for sequen of entities 
# input of sequences of X,O,S,-,P between 1 and 5
sent_input = Input(shape=(100,), dtype='int32', name='sent_input')

# embedding layer encodes the input into sequences of 300-dimenstional vectors. 
x = Embedding(output_dim=300, input_dim=5, input_length=500)(sent_input)

# add a convolutiaon 1D layer
x = Convolution1D(nb_filter=nb_filter, filter_length = filter_length, border_mode='valid', 
            activation='relu', subsample_length=1)(x)

# add max pooling layers
x = MaxPooling1D(pool_length=pool_length)(x)
x = Dropout(dropout_ratio)(x)
x = Dense(hidden_size, activation='relu')(x)
x = Dropout(dropout_ratio)(x)

# add latent cohernece score
out_x = Dense(1, activation='linear')(x)
shared_cnn = Model(sent_input, out_x)


# Inputs of pos and neg document
pos_input = Input(shape=(100,), dtype='int32')
neg_input = Input(shape=(100,), dtype='int32')

# the shared cnn model will share eveything
pos_branch = shared_cnn(pos_input)
neg_branch = shared_cnn(neg_input)

# adding latent coherence score
#pos_model = Dense(1, activation='linear')(pos_branch)
#neg_model = Dense(1, activation='linear')(neg_branch)

concatenated = merge([pos_branch, neg_branch], mode='concat')
# output by two latent coherence score
#predictions = Dense(2, activation='relu')(concatenated)

final_model = Model([pos_input, neg_input], concatenated)

final_model.compile(loss='ranking_loss', optimizer='adam')







