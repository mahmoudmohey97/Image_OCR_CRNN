import cv2
import numpy as np

from keras.utils import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

CHARACTERS = "اآأدذجحخهعغفقثصضطكمنتلبيسشظزوةىلارؤءئ "

def create_model_architecture(shape=(32,128,1)):
    inputs = Input(shape=shape)

    conv_1 = Conv2D(128, (3,3), activation = 'relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
    batch_norm_3 = BatchNormalization()(conv_3) #modified

    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(batch_norm_3)    
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)
    pool_5 = MaxPool2D(pool_size=(2, 1))(batch_norm_5)

    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_5)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(64, return_sequences=True, dropout = 0.2))(blstm_1)

    outputs = Dense(len(CHARACTERS)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time
    model = Model(inputs, outputs)
    return model, inputs, outputs

# Defining the CTC loss.
def ctc_lambda_function(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_layer(inputs, outputs, labels, input_length, label_length):
    # CTC layer declaration using lambda
    loss_out = Lambda(ctc_lambda_function, output_shape=(1,), name='ctc')([outputs, labels, 
                                                                            input_length,
                                                                           label_length])
    
    # Add CTC layer to train the model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    return model

def ctc_decoder(prediction):
    # use CTC decoder
    decoded_output = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                             greedy=True)[0][0])
    return decoded_output