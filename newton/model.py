import numpy as np  
# Keras
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.convolutional import Convolution2D, MaxPooling2D  
from keras.optimizers import SGD  
from keras.utils import np_utils 

def build_model(SGD_cache, Conv_cache, pool_side, shape, SUBJECT_NUM, EPOCH, BATCH_SIZE):
    # unpack parameters
    sgd_lr, sgd_decay, sgd_momentum = SGD_cache
    conv_side, filter1, filter2 = Conv_cache
    h, w = shape

    model = Sequential()

    print 'Building: conv1 - tanh - maxpooling'
    model.add(Convolution2D(filter1, conv_side, conv_side,
        border_mode='same', 
        subsample = (2, 2), dim_ordering='tf', 
        input_shape=(h, w, 1)))
    # print model.output_shape
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(pool_side, pool_side)))

    print 'Building: conv2 - tanh - maxpooling'
    model.add(Convolution2D(filter2, conv_side, conv_side))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(pool_side, pool_side)))  
    # model.add(Dropout(0.25))  

    print 'Building: flat - dense - tanh - dense - softmax'
    model.add(Flatten())  
    model.add(Dense(1000)) #Full connection  
    model.add(Activation('tanh'))  
    # model.add(Dropout(0.5))  
    model.add(Dense(SUBJECT_NUM))  
    model.add(Activation('softmax')) 

    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def train(model, arg_X, arg_y):
    pass