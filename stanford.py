# Stanford cs231n
import sys
import numpy as np
import cs231n
import attfaces

SUBJECT_NUM = 40
FACE_NUM = 10

X, y = attfaces.load_subjects(arg_flat = False)

def all_try():
    train, test = cs231n.variable.initialize_variables_entire(X, y)

    print 'training size:' + str(train['faces'].shape)
    print 'testing size: ' + str(test['faces'].shape)

    conv_params = {'stride':2, 'pad':1}
    pool_params = {'pool_height':2, 'pool_width':2, 'stride':2}
    params = cs231n.variable.initialize_parameters()
    
    model = cs231n.nn.ConvNet(params, 1e-1)

    bat = 1
    for i in xrange(2):
        for j in xrange(360/bat-1):
            loss, grads = model.loss(train['faces'][j*bat:j*bat+bat], train['label'][j*bat:j*bat+bat])
            print loss

    loss, grads = model.loss(test['faces'], test['label'])
    print loss

def exit_py():
    sys.exit()

func = {'0':all_try, 'exit':exit_py}

hint = open('./cs231n/help.txt').read()

if __name__ == '__main__':
    while(True):
        print hint
        case = input('cs231n > ')
        if case in func.keys():
            func[case]()