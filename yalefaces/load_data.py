import numpy as np

SUBJECT_NUM = 30
FACE_NUM = 15

def load_subjects(arg_flat = False, arg_onehot = False):
    print 'loading numpy array from ./yalefaces/dataset.npy ...'
    faces = np.load('./yalefaces/dataset.npy')
    label = np.arange(SUBJECT_NUM)

    if arg_flat is True:
        tmp = []
        for i in xrange(SUBJECT_NUM):
            subj = []
            for j in xrange(FACE_NUM):
                subj.append(faces[i][j].flatten())
            tmp.append(subj)
        faces = np.array(tmp)
    
    if arg_onehot is True:
        tmp = []
        for i in xrange(len(label)):
            pass

    return faces, label
