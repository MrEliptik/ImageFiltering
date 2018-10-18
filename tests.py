import filters as fltr
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
import imageio
from scipy import signal

from skimage import io

def Test2DConvolution():
    m1 = np.random.rand(5,5)
    m2 = fltr.Filter("box", 3)
    result1 = fltr.convolution2D(m1, m2.matrixFilter)
    # Same size output and zero padding
    result2 = signal.convolve2d(m1, m2.matrixFilter, mode='same', boundary='fill')
    print(result1)
    print(result2)
    if(result1.all() == result2.all()):
        print("Convolution test 1 passed !")
    else:
        print("Convolution test 1 failed !")

    m1 = np.random.rand(200,300)
    m2 = fltr.Filter("box", 7)
    result1 = fltr.convolution2D(m1, m2.matrixFilter)
    # Same size output and zero padding
    result2 = signal.convolve2d(m1, m2.matrixFilter, mode='same', boundary='fill')
    print(result1)
    print(result2)
    if(result1.all() == result2.all()):
        print("Convolution test 2 passed !")
    else:
        print("Convolution test 2 failed !")

    m1 = np.random.rand(200,300)
    m2 = fltr.Filter("box", 31)
    result1 = fltr.convolution2D(m1, m2.matrixFilter)
    # Same size output and zero padding
    result2 = signal.convolve2d(m1, m2.matrixFilter, mode='same', boundary='fill')
    print(result1)
    print(result2)
    if(result1.all() == result2.all()):
        print("Convolution test 3 passed !")
    else:
        print("Convolution test 3 failed !")

def TestBoxFilter():

    img_path = "stop_1_no_alpha.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(1, 4, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    img = imageio.imread(img_path)

    # Create filter 
    filtr = fltr.Filter("box", 3)

    # Filter it
    img_filtered = filtr.filterRGB(img)
    fig.add_subplot(1, 4, 2).title.set_text('Box 3x3')
    plt.axis('off')
    plt.imshow(img_filtered) 

    # Create filter 
    filtr = fltr.Filter("box", 5)

    # Filter it
    img_filtered = filtr.filterRGB(img)
    fig.add_subplot(1, 4, 3).title.set_text('Box 5x5')
    plt.axis('off')
    plt.imshow(img_filtered) 

     # Create filterRGB 
    filtr = fltr.Filter("box", 31)

    # Filter it
    img_filtered = filtr.filterRGB(img)
    fig.add_subplot(1, 4, 4).title.set_text('Box 31x31')
    plt.axis('off')
    plt.imshow(img_filtered) 

    # Show results
    plt.show()

def TestGaussianFilter():
    img_path = "stop_1_no_alpha.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    print(img.shape)
    fig.add_subplot(1, 4, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    img = imageio.imread(img_path)

    # Create filter 
    filtr = fltr.Filter("gaussian", 3, 1)

    # Filter it
    img_filtered = filtr.filterRGB(img)
    fig.add_subplot(1, 4, 2).title.set_text('Gaussian 3x3 sig=1')
    plt.axis('off')
    plt.imshow(img_filtered) 

    # Create filter 
    filtr = fltr.Filter("gaussian", 9, 1)

    # Filter it
    img_filtered = filtr.filterRGB(img)
    fig.add_subplot(1, 4, 3).title.set_text('Gaussian 9x9 sig=1')
    plt.axis('off')
    plt.imshow(img_filtered) 

     # Create filter 
    filtr = fltr.Filter("gaussian", 31, 1)

    # Filter it
    img_filtered = filtr.filterRGB(img)
    fig.add_subplot(1, 4, 4).title.set_text('Gaussian 31x31 sig=1')
    plt.axis('off')
    plt.imshow(img_filtered) 

    # Show results
    plt.show()

def TestEdgeFilter():
    img_path = "stop_1_no_alpha.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    print(img.shape)
    fig.add_subplot(2, 4, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    img = io.imread(img_path, as_gray=True)

    # Create filter 
    filtr = fltr.Filter("left_sobel", 3)

    # Filter it
    img_filtered = filtr.filterGrayscale(img)
    fig.add_subplot(2, 4, 2).title.set_text('Left sobel filter 3x3')
    plt.axis('off')
    plt.imshow(img_filtered, cmap=plt.cm.gray) 

    # Create filter 
    filtr = fltr.Filter("right_sobel", 3)

    # Filter it
    img_filtered = filtr.filterGrayscale(img)
    fig.add_subplot(2, 4, 3).title.set_text('Right sobel filter 3x3')
    plt.axis('off')
    plt.imshow(img_filtered, cmap=plt.cm.gray) 

    # Create filter 
    filtr = fltr.Filter("bottom_sobel", 3)

    # Filter it
    img_filtered = filtr.filterGrayscale(img)
    fig.add_subplot(2, 4, 4).title.set_text('Bottom sobel filter 3x3')
    plt.axis('off')
    plt.imshow(img_filtered, cmap=plt.cm.gray) 

    # Create filter 
    filtr = fltr.Filter("top_sobel", 3)

    # Filter it
    img_filtered = filtr.filterGrayscale(img)
    fig.add_subplot(2, 4, 5).title.set_text('Top sobel filter 3x3')
    plt.axis('off')
    plt.imshow(img_filtered, cmap=plt.cm.gray)

    
    # Create filter 
    filtr_top = fltr.Filter("top_sobel", 3)
    filtr_bottom = fltr.Filter("bottom_sobel", 3)
    filtr_left = fltr.Filter("left_sobel", 3)
    filtr_right = fltr.Filter("right_sobel", 3)

    # Filter it
    img_filtered = filtr_top.filterGrayscale(img)
    img_filtered = filtr_bottom.filterGrayscale(img_filtered)
    img_filtered = filtr_left.filterGrayscale(img_filtered)
    img_filtered = filtr_right.filterGrayscale(img_filtered)
 
    fig.add_subplot(2, 4, 6).title.set_text('All sobel filter 3x3')
    plt.axis('off')
    plt.imshow(img_filtered, cmap=plt.cm.gray)

    # Create filter 
    filter_gauss = fltr.Filter("gaussian", 5, 0.8)
    filtr_top = fltr.Filter("top_sobel", 3)
    filtr_bottom = fltr.Filter("bottom_sobel", 3)
    filtr_left = fltr.Filter("left_sobel", 3)
    filtr_right = fltr.Filter("right_sobel", 3)

    # Filter it
    img_filtered = filtr_top.filterGrayscale(img)
    img_filtered = filtr_top.filterGrayscale(img_filtered)
    img_filtered = filtr_bottom.filterGrayscale(img_filtered)
    img_filtered = filtr_left.filterGrayscale(img_filtered)
    img_filtered = filtr_right.filterGrayscale(img_filtered)
 
    fig.add_subplot(2, 4, 7).title.set_text('Gauss + All sobel filter 3x3')
    plt.axis('off')
    plt.imshow(img_filtered, cmap=plt.cm.gray)

    # Show results
    plt.show()

def TestGrayscaleConversion():
    img_path = "stop_1_no_alpha.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(2, 4, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    img = io.imread(img_path)

    #Convert to grayscale
    img_grayscale = fltr.RGBtoGrayscale(img)

    fig.add_subplot(2, 4, 2).title.set_text('Grayscale conversion')
    plt.axis('off')
    plt.imshow(img_grayscale, cmap=plt.cm.gray) 

    # Show results
    plt.show()

def TestStopSignFilter():
    img_path = "stop_1_no_alpha.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(2, 4, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    stop_photo = io.imread(img_path, as_gray=True)

    img_path = "stop_sign_only.png"

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(2, 4, 2).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    stop_only = io.imread(img_path, as_gray=True)

    result = fltr.convolution2D(stop_photo, stop_only)

    fig.add_subplot(2, 4, 3).title.set_text('Grayscale result')
    plt.axis('off')
    plt.imshow(result, cmap=plt.cm.gray) 

    # Create filter 
    filtr = fltr.Filter("top_sobel", 3)
    stop_filtered = filtr.filterGrayscale(stop_only)

    fig.add_subplot(2, 4, 4).title.set_text('Edge result')
    plt.axis('off')
    plt.imshow(stop_filtered, cmap=plt.cm.gray) 

    # Show results
    plt.show()

def main():
    
    #Test2DConvolution()

    #TestBoxFilter()

    #TestGaussianFilter()   

    #TestEdgeFilter()

    #TestGrayscaleConversion()

    TestStopSignFilter()

if __name__ == '__main__':
    main()