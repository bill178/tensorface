import numpy as np
import PIL.Image as Image

SUBJECT_NUM = 40
FACE_NUM = 10

def save_subjects(arg_size = None):
    all_array = []
    for dir_idx in xrange(1, 1+SUBJECT_NUM):
        subject_array = []
        for file_idx in xrange(1, 1+FACE_NUM):
            filename = './s%d/%d.pgm' % (dir_idx, file_idx)
            print 'reading ./s%d/%d.pgm ...' % (dir_idx, file_idx)
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
    save_subjects()