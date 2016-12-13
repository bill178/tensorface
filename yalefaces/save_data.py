import numpy as np
import os,json
import PIL.Image as Image

SUBJECT_NUM = 30
FACE_NUM = 15

def write_json():
    subjs = {}
    for dirpath,dirlist,filelist in os.walk('./'):
        for file in filelist:
            if not((('pgm' in file) and ('bad' not in file)) or ('py' in file)):
                os.remove('%s/%s'%(dirpath,file))
    for dirpath,dirlist,filelist in os.walk('./'):
        if 'yale' in dirpath:
            i = int(dirpath.replace('./yaleB', ''))
            print '%s'%dirpath
            subjs[i] = filelist[0:FACE_NUM]
    jsonstr = json.dumps(subjs)
    jsonf = open('./dictionary.json', 'w')
    jsonf.write(jsonstr)

def save_subjects(subjs, arg_size = None):
    all_array = []
    for i in subjs.keys()[0:SUBJECT_NUM]:
        subject_array = []
        for j in xrange(0, FACE_NUM):
            filename = './yaleB%02d/%s' % (int(i),subjs[i][j])
            print 'reading %s ...' % filename
            im = Image.open(filename)
            im = im.convert("L")
            # resize to given size (if given)
            if (arg_size is not None):
                im = im.resize(arg_size, Image.ANTIALIAS)
            subject_array.append(np.asarray(im, dtype=np.uint8))
        all_array.append(subject_array)
    all_array = np.array(all_array)
    np.save('./dataset.npy', all_array)
    print all_array.shape
    print 'saving subjects into ./dataset.npy ...'


if __name__ == "__main__":
    # subjs = write_json()
    with open('./dictionary.json', 'r') as jsonf:
        subjs = json.load(jsonf)
    save_subjects(subjs,(168,192))
