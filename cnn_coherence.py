from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = -K.sigmoid(pos-neg) # use loss = K.maximum(1.0 + neg - pos, 0.0) if you want to use margin ranking loss
return K.mean(loss) + 0 * y_true


#hyper parameres
nb_filter = 150
filter_length = 3
pool_length = 4
dropout_ratio = 0.5
hiden_size = 250

# first, define a CNN model for sequen of entities §
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
x = Dense(hiden_size, activation='relu')(x)
x = Dropout(dropout_ratio)(x)

# add latent cohernece score
# out_x = Dense(1, activation='linear')(x)


shared_cnn = Model(sent_input, out_x)


# Inputs of pos and neg document
pos_input = Input(shape=(100,), dtype='int32')
neg_input = Input(shape=(100,), dtype='int32')

# the shared cnn model will share eveything
pos_branch = shared_cnn(pos_input)
neg_branch = shared_cnn(neg_input)

# adding latent coherence score
pos_model = Dense(1, activation='linear')(pos_model)
neg_model = Dense(1, activation='linear')(neg_model)

concatenated = merge([pos_model, neg_model], mode='concat')










