import numpy as np 
import tensorflow as tf 

def initialize_variables_entire(arg_X, arg_y, onehot = False):
    # arg_X.shape = (SUBJECT_NUM, FACE_NUM, d)
    (sn, fn, h, w) = arg_X.shape
    assert len(arg_y) == sn

    train_list, test_list = [], []
    train_y, test_y = [], []
    for i in xrange(sn):
        test_list.append(arg_X[i, 0])
        if onehot is True:
            tmp_y = np.zeros(sn)
            tmp_y[int(arg_y[i])] = 1
            test_y.append(tmp_y)
        else:
            test_y.append(arg_y[i])
        for j in xrange(1, fn):
            train_list.append(arg_X[i, j])
            if onehot is True:
                tmp_y = np.zeros(sn)
                tmp_y[int(arg_y[i])] = 1
                train_y.append(tmp_y)
            else:
                train_y.append(arg_y[i])
    
    train, test = {}, {}

    train['faces'] = np.array(train_list).reshape(sn*(fn-1), h, w, 1)
    train['label'] = np.array(train_y)

    test['faces'] = np.array(test_list).reshape(sn, h, w, 1)
    test['label'] = np.array(test_y)
    
    return train, test
