from scipy import misc
from scipy import signal
import numpy as np  
import scipy.stats as st
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
import imageio
 

class Filter:
    def __init__(self, type, size, sigma=1):
        self.type = type
        self.size = size
        self.matrixFilter = np.empty([self.size, self.size])
        self.createMatrixFilter(sigma)

    def createMatrixFilter(self, sigma=1):
        if(self.type == "box"):
            self.matrixFilter.fill(1/(self.size*self.size))
        elif(self.type == "gaussian"):        
            self.matrixFilter = gkern(self.size, sigma)
        elif(self.type == "sharpen"):
            pass
        elif(self.type == "left_sobel"):
            self.matrixFilter = np.array([[1, 0, -1], \
                                          [2, 0, -2], \
                                          [1, 0, -1]])
        elif(self.type == "right_sobel"):
            self.matrixFilter = np.array([[-1, 0, 1], \
                                          [-2, 0, 2], \
                                          [-1, 0, 1]])
        elif(self.type == "bottom_sobel"):
            self.matrixFilter = np.array([[-1, -2, -1], \
                                          [0, 0, 0], \
                                          [1, 2, 1]])
        elif(self.type == "top_sobel"):
            self.matrixFilter = np.array([[1, 2, 1], \
                                          [0, 0, 0], \
                                          [-1, -2, -1]])

    
    def filterRGB(self, img_matrix):
        ims = []
        for d in range(3):
            im_conv_d = convolution2D(img_matrix[:,:,d], self.matrixFilter)
            ims.append(im_conv_d)

        im_conv = np.stack(ims, axis=2).astype("uint8")
        return im_conv

    def filterGrayscale(self, img_matrix):
        im_conv_d = convolution2D(img_matrix, self.matrixFilter)
        return im_conv_d

def threshold(img_matrix, threshold=0.5, keepvalue=False):
    result = np.empty([img_matrix.shape[0], img_matrix.shape[1]])
    # We assume the img is grayscale
    # Loop over every pixel of the image
    for column in range(img_matrix.shape[1]):     
        for row in range(img_matrix.shape[0]):
            if(img_matrix[row, column] > threshold):
                if(keepvalue):
                    result[row, column] = img_matrix[row, column]
                else:
                    result[row, column] = 1
            else:
                result[row, column] = 0
    return result

def squareTracing(img_matrix):
    contour_points = []
    x = None
    y = None
    start_x = None
    start_y = None
    orientation = "down"

    for column in range(img_matrix.shape[1]):     
        for row in range(img_matrix.shape[0]):
            if(img_matrix[row, column] == 1):
                # White pixel is found, and set as starting point
                contour_points.append((row, column))
                start_x, start_y = (row, column)
    
    # If not initiliazed, we start with the start pixel
    x = start_x
    y = start_y
    # And go left because current pixel is white
    x = start_x - 1
    while(x != start_x & y != start_y):
        # If current pixel is white, we go 'left' with
        # respect to the current orientations
        if((img_matrix[x, y] == 1) & (orientation == "down")):
            # Add it to the list
            contour_points.append(img_matrix[x, y])
            # Go right
            x = x + 1
        if((img_matrix[x, y] == 1) & (orientation == "up")):
            contour_points.append(img_matrix[x, y])
            # Go left
            x = x - 1
        if((img_matrix[x, y] == 1) & (orientation == "right")):
            contour_points.append(img_matrix[x, y])
            # Go up
            y = y - 1
        if((img_matrix[x, y] == 1) & (orientation == "left")):
            contour_points.append(img_matrix[x, y])
            # Go down
            y = y + 1

        # If current pixel is black, go 'right'.
        # Going right with respect to the current orientation
        if((img_matrix[x, y] == 0) & (orientation == "down")):
            # Go left
            x = x - 1
        if((img_matrix[x, y] == 0) & (orientation == "up")):
            # Go right
            x = x + 1
        if((img_matrix[x, y] == 0) & (orientation == "right")):
            # Go down
            y = y + 1
        if((img_matrix[x, y] == 0) & (orientation == "left")):
            # Go go up
            y = y - 1
        
    return contour_points

def convolution2D(m1, m2):
    result = np.empty([m1.shape[0], m1.shape[1]])

    row_padding = int((m2.shape[0]-1))
    column_padding = int((m2.shape[1]-1))
    conv_x_idx = int(row_padding/2)
    conv_y_idx = int(column_padding/2)
    # NOT IDEAL : Add zero padding, to have to same output size as m1
    m1_padded = np.zeros((m1.shape[0] + row_padding, m1.shape[1] + column_padding))   
    m1_padded[conv_x_idx:-conv_x_idx, conv_y_idx:-conv_y_idx] = m1

    # Loop over every pixel of the image
    for column in range(m1.shape[1]):     
        for row in range(m1.shape[0]):
            # element-wise multiplication of the kernel and the image
            result[row,column]=(m2*m1_padded[row:row+m2.shape[0],column:column+m2.shape[1]]).sum() 
    return result

def gkern(kernlen=5, sigma=1):
    #Returns a 2D Gaussian kernel array.
    interval = (2*sigma+1.)/(kernlen)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def RGBtoGrayscale(rgb_matrix):
    grayscale_matrix = np.empty([rgb_matrix.shape[0], rgb_matrix.shape[1]])
    # Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    for j in range(rgb_matrix.shape[0]):
        for i in range(rgb_matrix.shape[1]):
            grayscale_matrix[j,i] = 0.2126 * rgb_matrix[j,i,0] + 0.7152 * rgb_matrix[j,i,1] + 0.0722 * rgb_matrix[j,i,2]
    return grayscale_matrix

def filter2D(big_matrix, filter_matrix):
    #Convolves im with window, over all three colour channels
    ims = []
    for d in range(3):
        im_conv_d = signal.convolve2d(big_matrix[:,:,d], filter_matrix, mode="same", boundary="symm")
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    return im_conv

