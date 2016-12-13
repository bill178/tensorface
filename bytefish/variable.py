import numpy as np

def initialize_variables_entire(arg_X, arg_y):
    # arg_X.shape = (SUBJECT_NUM, FACE_NUM, d)
    (sn, fn, d) = arg_X.shape
    assert len(arg_y) == sn

    train_list, test_list = [], []
    train_y, test_y = [], []
    for i in xrange(sn):
        test_list.append(arg_X[i, 0])
        test_y.append(arg_y[i])
        for j in xrange(1, fn):
            train_list.append(arg_X[i, j])
            train_y.append(arg_y[i])
    
    train, test = {}, {}

    train['faces'] = np.array(train_list)
    train['label'] = np.array(train_y)

    test['faces'] = np.array(test_list)
    test['label'] = np.array(test_y)

    return train, test

# leave one test
def initialize_variables_single(arg_X, arg_y, test_num):
    # arg_X.shape = (SUBJECT_NUM, FACE_NUM, d)
    # arg_y.shape = (SUBJECT_NUM, )
    # test_num == -1 means make the entire set as training
    (sn, fn, d) = arg_X.shape
    assert len(arg_y) == sn
    assert test_num <= sn

    train_list, test_list = [], []
    train_y, test_y = [], []
    for i in xrange(sn):
        if i != test_num:
            for j in xrange(fn):
                train_list.append(arg_X[i, j])
                train_y.append(arg_y[i])
        else:
            test_list = arg_X[i, 0]
            test_y.append(arg_y[i])
            for j in xrange(1, fn):
                train_list.append(arg_X[i, j])
                train_y.append(arg_y[i])

    train, test = {}, {}

    train['faces'] = np.array(train_list)
    train['label'] = np.array(train_y)

    test['faces'] = np.array(test_list)
    test['label'] = np.array(test_y)

    return train, test
 