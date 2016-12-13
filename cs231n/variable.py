import numpy as np

def initialize_variables_entire(arg_X, arg_y):
    # arg_X.shape = (SUBJECT_NUM, FACE_NUM, d)
    (sn, fn, h, w) = arg_X.shape
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

    train['faces'] = np.array(train_list).reshape(sn*(fn-1), 1, h, w)
    train['label'] = np.array(train_y)

    test['faces'] = np.array(test_list).reshape(sn, 1, h, w)
    test['label'] = np.array(test_y)

    return train, test

def initialize_parameters(input_shape = (1, 112, 92), num_fileters = 32,
                fileter_size = 7, hidden_dim = 100, num_class = 40,
                weight_scale = 1e-3, dtype = np.float32):
    # set param is used to save parameters    
    params = {}

    params['W1'] = np.random.normal(scale = weight_scale, size = (num_fileters, input_shape[0], fileter_size, fileter_size))
    W2_row_size = num_fileters * (input_shape[1]//2) * (input_shape[2]//2)
    params['W2'] = np.random.normal(scale = weight_scale, size = (W2_row_size, hidden_dim))
    params['W3'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_class))

    params['b1'] = np.zeros(num_fileters)
    params['b2'] = np.zeros(hidden_dim)
    params['b3'] = np.zeros(num_class)

    for name, value in params.iteritems():
        params[name] = value.astype(dtype)

    print 'all variables initialized'
    return params

