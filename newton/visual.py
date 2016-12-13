import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import newton
# Keras
from keras.models import Sequential  
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.convolutional import Convolution2D, MaxPooling2D  
from keras.optimizers import SGD  
from keras.utils import np_utils

def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center') 
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        
def imsave(image, title="", filename=None):
    plt.figure()
    plt.imshow(np.asarray(image))
    plt.title(title, create_font('Tahoma',10))
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1]. 
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def draw_img(arg_X, arg_title, arg_sptitle, arg_filename):
    (N, H, W, D) = arg_X.shape
    conv = []
    maxcols = D
    if D > 4:
        maxcols = 4
    for i in xrange(2):
        for j in xrange(maxcols):
            conv.append(normalize(arg_X[i,:,:,j], 0, 255))
    subplot(title=arg_title, images=conv, rows = 2, cols = maxcols, sptitle=arg_sptitle, colormap=cm.jet, filename=arg_filename)

SUBJECT_NUM = 40
FACE_NUM = 10
EPOCH = 40
BATCH_SIZE = 40

def drawconv1(model, arg_X, SGD_cache, Conv_cache, pool_side, shape):
    print 'drawing conv1 ...'
    sgd_lr, sgd_decay, sgd_momentum = SGD_cache
    conv_side, filter1, filter2 = Conv_cache
    h, w = shape
    model2 = Sequential()
    model2.add(Convolution2D(filter1, conv_side, conv_side,
        border_mode='same', weights = model.layers[0].get_weights(),
        subsample = (2, 2), dim_ordering='tf', 
        input_shape=(112, 92, 1)))
    act = model2.predict(arg_X)
    print act.shape
    draw_img(act,"Conv Layer1","Conv 1 Face","./image/conv1.png")
    del model2

def drawpool1(model, arg_X, SGD_cache, Conv_cache, pool_side, shape):
    print 'drawing pooling1 ...'
    sgd_lr, sgd_decay, sgd_momentum = SGD_cache
    conv_side, filter1, filter2 = Conv_cache
    h, w = shape
    model2 = Sequential()
    model2.add(Convolution2D(filter1, conv_side, conv_side,
        border_mode='same', weights = model.layers[0].get_weights(),
        subsample = (2, 2), dim_ordering='tf', 
        input_shape=(112, 92, 1)))
    model2.add(Activation('tanh'))
    model2.add(MaxPooling2D(pool_size=(pool_side, pool_side)))
    act = model2.predict(arg_X)
    print act.shape
    draw_img(act,"Max Pooling 1","Pooling 1 Face","./image/pool1.png")
    del model2

def drawconv2(model, arg_X, SGD_cache, Conv_cache, pool_side, shape):
    print 'drawing conv2 ...'
    sgd_lr, sgd_decay, sgd_momentum = SGD_cache
    conv_side, filter1, filter2 = Conv_cache
    h, w = shape
    model2 = Sequential()
    model2.add(Convolution2D(filter1, conv_side, conv_side,
        border_mode='same', weights = model.layers[0].get_weights(),
        subsample = (2, 2), dim_ordering='tf', 
        input_shape=(112, 92, 1)))
    model2.add(Activation('tanh'))
    model2.add(MaxPooling2D(pool_size=(pool_side, pool_side)))
    model2.add(Convolution2D(filter2, conv_side, conv_side)) 
    act = model2.predict(arg_X)
    print act.shape
    draw_img(act,"Conv Layer2","Conv 2 Face","./image/conv2.png")
    del model2

def drawpool2(model, arg_X, SGD_cache, Conv_cache, pool_side, shape):
    print 'drawing conv2 ...'
    sgd_lr, sgd_decay, sgd_momentum = SGD_cache
    conv_side, filter1, filter2 = Conv_cache
    h, w = shape
    model2 = Sequential()
    model2.add(Convolution2D(filter1, conv_side, conv_side,
        border_mode='same', weights = model.layers[0].get_weights(),
        subsample = (2, 2), dim_ordering='tf', 
        input_shape=(112, 92, 1)))
    model2.add(Activation('tanh'))
    model2.add(MaxPooling2D(pool_size=(pool_side, pool_side)))
    model2.add(Convolution2D(filter2, conv_side, conv_side))
    model2.add(Activation('tanh'))  
    model2.add(MaxPooling2D(pool_size=(pool_side, pool_side)))  
    act = model2.predict(arg_X)
    print act.shape
    draw_img(act,"Max Pooling 2","Pooling 2 Face","./image/pool2.png")
    del model2

def draw_model(model, arg_X, SGD_cache, Conv_cache, pool_side, shape):
    drawconv1(model, arg_X, SGD_cache, Conv_cache, pool_side, shape)
    drawpool1(model, arg_X, SGD_cache, Conv_cache, pool_side, shape)
    drawconv2(model, arg_X, SGD_cache, Conv_cache, pool_side, shape)
    drawpool2(model, arg_X, SGD_cache, Conv_cache, pool_side, shape)
