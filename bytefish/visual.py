import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import bytefish

def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center') 
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
        
def imsave(image, title="", filename=None):
    plt.figure()
    plt.imshow(np.asarray(image))
    plt.title(title, create_font('Tahoma',10))
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1]. 
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def draw_eigenfaces(arg_X, arg_shape):
    # perform a full pca
    W, mu, cache = bytefish.subspace.pca(arg_X)
    # turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in xrange(min(len(arg_X), 16)):
        e = W[:,i].reshape(arg_shape)
        E.append(normalize(e,0,255))
    subplot(title="Eigenfaces Yale Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="./image/eigenfaces.png")

    # reconstruction steps
    steps=[i for i in xrange(10, min(len(arg_X), 320), 20)]
    E = []
    for i in xrange(min(len(steps), 16)):
        numEvs = steps[i]
        P = bytefish.subspace.project(W[:,0:numEvs], arg_X[0].reshape(1,-1), mu)
        R = bytefish.subspace.reconstruct(W[:,0:numEvs], P, mu)
        # reshape and append to plots
        R = R.reshape(arg_shape)
        E.append(normalize(R,0,255))
    subplot(title="Reconstruction Yale Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename="./image/eigenvectors.png")
