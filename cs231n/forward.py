import numpy as np

def conv_relu_pool_forward(x, w1, b1, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    conv_out, conv_cache = conv_forward_fast(x, w1, b1, conv_param)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = maxpool_forward(relu_out, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return pool_out, cache

def full_relu_forward(x, w2, b2):
    """
    Convenience layer that perorms an full transform followed by a ReLU

    Inputs:
    - x: Input to the full layer
    - w, b: Weights for the full layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    full_out, full_cache = full_forward(x, w2, b2)
    relu_out, relu_cache = relu_forward(full_out)
    cache = (full_cache, relu_cache)
    return relu_out, cache

def full_forward(x, w, b):
    """
    Computes the forward pass for an full (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M) 
    - cache: (x, w, b)
    """
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_flat = x.reshape(N, D)

    out = np.dot(x_flat, w) + b
    cache = (x, w, b)
    return out, cache

def conv_forward(x, w1, b1, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    (N, C, H, W) = x.shape
    (F, CC, HH, WW) = w1.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    assert C == CC, '! illegal input depth'
    assert (H + 2 * pad - HH) % stride == 0, '! illegal widith combination'
    assert (W + 2 * pad - WW) % stride == 0, '! illegal height combination'

    H += 2 * pad
    W += 2 * pad
    step_h = 1 + (H - HH) / stride
    step_w = 1 + (W - WW) / stride

    # padding
    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode = 'constant', constant_values = 0)

    # divide padded into blocks
    x_blocks = np.zeros((N, C, step_h, step_w, HH, WW))
    for i in xrange(N):
        this_img = x_padded[i]
        for j in xrange(C):
            this_channel = this_img[j]
            for k in xrange(step_h):
                row_begin = k * stride
                for l in xrange(step_w):
                    col_begin = l * stride
                    x_blocks[i, j, k, l] = this_channel[row_begin:row_begin+HH, col_begin:col_begin+HH]
    

    # convolutional calculation
    out = np.zeros((N, F, step_h, step_w))
    for i in xrange(N):
        for j in xrange(F):
            for k in xrange(step_h):
                for l in xrange(step_w):
                    out[i, j, k, l] = 0
                    for m in xrange(C):
                        out[i, j, k, l] += np.sum(x_blocks[i, m, k, l] * w1[j, m])
                    out[i, j, k, l] += b1[j]
    cache = (x, w1, b1, conv_param)
    return out, cache

def conv_forward_fast(x, w, b, conv_param):
    '''
    fast version from Stanford cs231n
    '''
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Check dimensions
    assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) / stride + 1
    out_w = (W - WW) / stride + 1

    # Perform an im2col operation by picking clever strides
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # Now all our convolutions are a big matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    # Reshape the output
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    # Be nice and return a contiguous array
    # The old version of conv_forward_fast doesn't do this, so for a fair
    # comparison we won't either
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache

def maxpool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions
    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    (N, C, H, W) = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    step_h = 1 + (H - pool_h) / stride
    step_w = 1 + (W - pool_w) / stride

    out = np.zeros((N, C, step_h, step_w))

    for i in xrange(N):
        this_img = x[i]
        for j in xrange(C):
            this_channel = this_img[j]
            for k in xrange(step_h):
                row_begin = k * stride
                for l in xrange(step_w):
                    col_begin = l * stride
                    out[i, j, k, l] = np.max(this_channel[row_begin:row_begin+pool_h, col_begin:col_begin+pool_w])
    cache = (x, pool_param)

    return out, cache
