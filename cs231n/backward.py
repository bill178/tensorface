import numpy as np

import pyximport
pyximport.install()
from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython

def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    dout_relu = maxpool_backward(dout, pool_cache)
    dout_conv = relu_backward(dout_relu, relu_cache)
    dx, dw, db = conv_backward_strides(dout_conv, conv_cache)
    return dx, dw, db

def full_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    full_cache, relu_cache = cache
    dout_full = relu_backward(dout, relu_cache)
    dx, dw, db = full_backward(dout_full, full_cache)
    return dx, dw, db

def full_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_flat = x.reshape(N, D)

    db = np.sum(dout, axis = 0)
    dw = x_flat.T.dot(dout)
    dx = dout.dot(w.T).reshape(x.shape)

    return dx, dw, db

def conv_backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
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
    # unpack
    dx, dw, db = None, None, None
    #############################################################################
    # Implement the convolutional backward pass.                                #
    # Inspired by                                                               #
    # https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/layers.py #
    #############################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    pad = conv_param['pad']
    stride = conv_param['stride']
    x_with_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)

    N, F, Hdout, Wdout = dout.shape

    H_out = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
    W_out = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']

    db = np.zeros((b.shape))
    for i in range(0, F):
        db[i] = np.sum(dout[:, i, :, :])

    dw = np.zeros((F, C, HH, WW))
    for i in range(0, F):
        for j in range(0, C):
            for k in range(0, HH):
                for l in range(0, WW):
                    dw[i, j, k, l] = np.sum(dout[:, i, :, :] * x_with_pad[:, j, k:k + Hdout * stride:stride, l:l + Wdout * stride:stride])

    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hdout):
                        for l in range(Wdout):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + pad - k * stride) < HH and (i + pad - k * stride) >= 0:
                                mask1[:, i + pad - k * stride, :] = 1.0
                            if (j + pad - l * stride) < WW and (j + pad - l * stride) >= 0:
                                mask2[:, :, j + pad - l * stride] = 1.0

                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked

    return dx, dw, db

def conv_backward_strides(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    out = np.maximum(0, x) # ReLu again
    out[out > 0] = 1
    dx = out * dout

    return dx

def maxpool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros((N, C, H, W))
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride

    for i in range(0, N):
        x_data = x[i]

        xx, yy = -1, -1
        for j in range(0, H-pool_height+1, stride):
            yy += 1
            for k in range(0, W-pool_width+1, stride):
                xx += 1
                x_rf = x_data[:, j:j+pool_height, k:k+pool_width]
                for l in range(0, C):
                    x_pool = x_rf[l]
                    mask = x_pool == np.max(x_pool)
                    dx[i, l, j:j+pool_height, k:k+pool_width] += dout[i, l, yy, xx] * mask

            xx = -1

    return dx