import numpy as np
import cs231n

class ConvNet(object):
    '''
    Reference see Stanford cs231n assignment, architecture:
    conv - relu - 2x2 max pool - full - relu - full - softmax
    '''
    def __init__(self, arg_params, reg = 0.0, dtype = np.float32):
        print 'initializing model convolutional net...'
        self.params = arg_params
        self.reg = reg
        self.dtype = dtype

    def loss(self, arg_X, arg_y = None):
        print 'convolutional neural net is running ...'
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # convolutional parameters
        fileter_size = W1.shape[2]
        conv_param = {'stride':1, 'pad':(fileter_size - 1)/2}
        
        # pooling parameters
        pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}

        print 'propagating forwards ...'
        scores = None
        out_1, cache_1 = cs231n.forward.conv_relu_pool_forward(arg_X, W1, b1, conv_param, pool_param)
        out_2, cache_2 = cs231n.forward.full_relu_forward(out_1, W2, b2)
        out_3, cache_3 = cs231n.forward.full_forward(out_2, W3, b3)
        scores = out_3

        if arg_y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = cs231n.softmax_loss(scores, arg_y)
        loss += sum(0.5 * self.reg * np.sum(W_tmp ** 2) for W_tmp in [W1, W2, W3])

        print 'propagating backwards ...'
        dx_3, grads['W3'], grads['b3'] = cs231n.backward.full_backward(dscores, cache_3)
        dx_2, grads['W2'], grads['b2'] = cs231n.backward.full_relu_backward(dx_3, cache_2)
        dx_1, grads['W1'], grads['b1'] = cs231n.backward.conv_relu_pool_backward(dx_2, cache_1)

        grads['W3'] += self.reg*self.params['W3']
        grads['W2'] += self.reg*self.params['W2']
        grads['W1'] += self.reg*self.params['W1']

        return loss, grads