from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import math
from PIL import Image
from theano import shared
import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    print ("sz shape : ", sz)
    print ("D shape : ", D.shape)
    f, axarr = plt.subplots(4, 4)

    for i in range(4):
        for j in range(4):
            Dij = D[:, (i*4)+j]
            #print ("Dij shape : ",Dij.shape)
            d = np.reshape(Dij , (sz,sz) )
            #plot(cij, Dij, n_blocks, X_mn, axarr[i, j])
            ax = axarr[i,j]
            ax.imshow(d , cmap=cm.Greys_r)

    f.savefig(imname.format(sz,sz))
    plt.close(f)


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''
    #print ("c shape : ",c.shape.eval())
    #print ("D shape : ",D.shape.eval())
    
    X = np.dot(c.T,D.T)
    print X_mn.shape
    print X.shape
    #X = X + np.repeat(X_mn.reshape(1, -1), X.shape[0], 0
    X = X.reshape((256,256))
    X = X + X_mn 
    ax.imshow(X, cmap=cm.Greys_r)


if __name__ == '__main__':
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''
    file_list = glob.glob("./jaffe" + '/*.tiff');
    file_list.sort()
    LENGTH = len(file_list)
    Ims = np.empty( (LENGTH,256,256) )
    count_image = 0;
    for np_name in file_list:
        Ims[count_image] = (Image.open(np_name))
        count_image = count_image + 1;

    Ims = Ims.reshape((LENGTH,65536))
    Ims = Ims.astype(np.float32) 
    X_mn = np.mean(Ims, 0)   
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    print ("X shape : ", X.shape)
    
    N = 16;
    grad_old = 0;
    Steps = 10;
    eta = 0.01;
    d = np.random.rand(X.shape[1],N)
    norm_d = np.linalg.norm(d)
    d = d/norm_d
    D = shared(d,borrow=True)
    lamda = shared(np.zeros((1,N)), broadcastable = (True,False))
    print ("entered while")
    
    for i in range(N):
        Di = D[ : , i]
        prod = T.dot(T.dot(T.dot( Di.T , X.T),X),Di) 
        
        sum_var = T.dot(T.dot(T.dot(Di.T , D*lamda) , D.T ), Di)
        cost = (prod - sum_var).norm(2)
        grad = T.grad(cost, Di)
        
        y = Di - (eta * grad)
        y=y/y.norm(2)
        update_D = T.set_subtensor(Di, y)
        f=theano.function([],y,updates=[(D,update_D)])
        t = 1;
        threshold = math.pow(1,-13)
        change = 10
        old_cost = math.pow(1,10)
        while (t <= Steps and change > threshold):
            
            t = t + 1
            z = f()
            
            change = np.absolute((old_cost - z).sum())
            

            old_cost = z
        lamdai=lamda[:,i]
        comput = T.dot(X,Di)
        update_lamda=T.set_subtensor(lamdai,T.dot(comput.T , comput))
        g=theano.function([],updates=[(lamda,update_lamda)])
        g()
        #print ("exit while")
        print t
        print lamdai.eval()
        #print D.get_value()
    #print ("X shape : ", X.shape)
    #print ("D shape : ", D.shape.eval())
    c = T.dot(X,D)
    c = c.T

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues
    
    
            

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''

    for i in range(0,200,10):
        plot_mul(c.eval(), D.eval(), i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D.eval(), 256, 'output/hw1b_top16_256.png')