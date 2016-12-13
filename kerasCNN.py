import sys,json
import numpy as np  
# use 'newton' module, thus named nwt.py
import newton 
import attfaces
import yalefaces
# Keras
from keras.models import load_model

DATASET = 'yale'

if DATASET == 'att':
    SUBJECT_NUM, FACE_NUM = 40, 10
    EPOCH, BATCH_SIZE = 40, 40
    X, y = attfaces.load_subjects(arg_flat = False)
    train, test = newton.variable.initialize_variables_entire(X, y, onehot = True)
elif DATASET == 'yale':
    SUBJECT_NUM, FACE_NUM = 30, 15
    EPOCH, BATCH_SIZE = 40, 40
    X, y = yalefaces.load_subjects(arg_flat = False)
    train, test = newton.variable.initialize_variables_entire(X, y, onehot = True)

SGD_cache = [0.005, 1e-6, 0.9]
Conv_cache = [4, 5, 10]

print train['faces'].shape

def save_mode():
    if DATASET == 'att':
        model = newton.model.build_model(SGD_cache, Conv_cache, 2, (112, 92), SUBJECT_NUM, EPOCH, BATCH_SIZE)
    elif DATASET == 'yale':
        model = newton.model.build_model(SGD_cache, Conv_cache, 2, (192, 168), SUBJECT_NUM, EPOCH, BATCH_SIZE)
    # training
    model.fit(train['faces'], train['label'], batch_size = BATCH_SIZE,
        nb_epoch = EPOCH, show_accuracy = True, verbose = 1)
    model.save('./newton/modelCNN.h5')
    model.save_weights('./newton/weightCNN.h5')
    del model

def all_try():
    model = load_model('./newton/modelCNN.h5')
    model.load_weights('./newton/weightCNN.h5')
    print 'Test results:'    
    # comparision
    classes = model.predict_classes(test['faces'], verbose = 0)
    print classes

def draw():
    model = load_model('./newton/modelCNN.h5')
    model.load_weights('./newton/weightCNN.h5')
    if DATASET == 'att':
        newton.visual.draw_model(model, train['faces'], SGD_cache, Conv_cache, 2, (112, 92))

def exit_py():
    sys.exit()

func = {'0':save_mode, '1':all_try, '2':draw, 'exit':exit_py}

hint = open('./newton/help.txt').read()

if __name__ == '__main__':
    while(True):
        print hint
        case = raw_input('newton > ')
        if case in func.keys():
            func[case]()