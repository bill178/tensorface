import sys
import numpy as np
import bytefish
import attfaces
import yalefaces

X, y = [],[]

DATASET = 'yale'

if DATASET == 'yale':
    SUBJECT_NUM = 30
    FACE_NUM = 15
    X, y = yalefaces.load_subjects(arg_flat = True)
elif DATASET == 'att':
    SUBJECT_NUM = 40
    FACE_NUM = 10
    X, y = attfaces.load_subjects(arg_flat = True)

def all_try():
    train, test = bytefish.variable.initialize_variables_entire(X, y)
    model = bytefish.model.EigenFace(train['faces'], train['label'])
    for i in xrange(len(test['faces'])):
        print "expected =", test['label'][i], "/", "predicted =", model.predict(test['faces'][i])

def leave_one():
    if DATASET == 'att':
        print 'Enter subject index in [0, 39]'
        sub_idx = input('eigenface - 1 > ')
        train, test = bytefish.variable.initialize_variables_single(X, y, sub_idx)
        model = bytefish.model.EigenFace(train['faces'], train['label'])
        print "expected =", test['label'][0], "/", "predicted =", model.predict(test['faces'])

def draw_face():
    train, test = bytefish.variable.initialize_variables_single(X, y, -1)
    if DATASET == 'att':
        bytefish.visual.draw_eigenfaces(train['faces'], (112, 92))
    elif DATASET == 'yale':
        bytefish.visual.draw_eigenfaces(train['faces'], (192, 168))

def exit_py():
    sys.exit()

func = {'exit': exit_py, '0':all_try, '1':leave_one, '2':draw_face}

hint = open('./bytefish/help.txt').read()

if __name__ == '__main__':
    while(True):
        print hint
        case = raw_input('eigenface > ')
        if case in func.keys():
            func[case]()