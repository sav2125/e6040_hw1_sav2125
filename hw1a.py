from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs
from theano.tensor.nnet.neighbours import neibs2images

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs, n_blocks):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    f, axarr = plt.subplots(3, 3)
    print ("axarr : ",axarr.shape)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
            Dij = D[:, :nc]
            plot(cij, Dij, n_blocks, X_mn, axarr[i, j])

    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

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
    

def plot(c, D, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
    X = np.dot(c.T,D.T)
    print X_mn.shape
    print X.shape
    X = X + np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)
    print X.shape
    size = 256/n_blocks;
    new_matrix = T.matrix('new_matrix');
    im_new = neibs2images(new_matrix, (size, size), (1,1,256,256) )
    # Theano function definition
    inv_window = theano.function([new_matrix], im_new, allow_input_downcast = True)
    # Function application
    im_new_val = inv_window(X)
    im_new_val = im_new_val.reshape(256,256)
    print ("im_new_val : ",im_new_val.shape)
    ax.imshow(im_new_val, cmap=cm.Greys_r)
    
    

def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    
    # my code for reading files start
    file_list = glob.glob("./jaffe" + '/*.tiff');
    file_list.sort()
    LENGTH = len(file_list)
    Ims = np.empty( (LENGTH,256,256) )
    count_image = 0;
    for np_name in file_list:
        Ims[count_image] = (Image.open(np_name))
        count_image = count_image + 1;
    #print(Ims)
    # my code for reading files end
    

    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''
        #my code starts
        images = theano.tensor.tensor4('images')
        neibs = images2neibs(images, neib_shape=(sz, sz))
        window_function = theano.function([images], neibs, allow_input_downcast=True)
        a,b,c = Ims.shape
        ims_val = np.reshape(Ims,(1, a,b,c))
        #neibs_val = window_function(ims_val)
        X = window_function(ims_val)
        #print ims_val.shape
        print X.shape

        #print neibs_val.shape
        #my code ends

        X_mn = np.mean(X, 0)
        print X_mn.shape
        a = X_mn.reshape(1,-1)
        print a.shape
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)
        

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        #my code starts
        X_Matrix_for_eigen = np.dot(X.T,X);
        print ("X_Matrix_for_eigen.shape : ",X_Matrix_for_eigen.shape)
        eigenValues,D = np.linalg.eig(X_Matrix_for_eigen)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        #print eigenValues
        D = D[:,idx]
        #print D.shape
        #my code ends

        c = np.dot(D.T, X.T)

        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape((sz, sz)),
                     num_coeffs=nc, n_blocks=int(256/sz))
        print ("D in main: ",D.shape)
        print sz
        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
