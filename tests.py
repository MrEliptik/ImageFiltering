import filters as fltr
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
import matplotlib.patches as patches
import imageio
from scipy import signal

from skimage.filters import gaussian
from skimage.segmentation import active_contour

from skimage import io
from skimage import feature

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

def TestCannyEdgeFilter():
    img_path = "stop_1.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(2, 4, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    stop_photo = io.imread(img_path, as_gray=True)

    fig.add_subplot(2, 4, 2).title.set_text('Grayscale result')
    plt.axis('off')
    plt.imshow(stop_photo, cmap=plt.cm.gray)

    # Threshold 
    thresholded = fltr.threshold(stop_photo, 0.8) #0.82
    fig.add_subplot(2, 4, 3).title.set_text('Threshold result')
    plt.axis('off')
    plt.imshow(thresholded, cmap=plt.cm.gray)
    
    # Canny edge
    edges1 = feature.canny(stop_photo, 1.5)
    fig.add_subplot(2, 4, 4).title.set_text('Canny edge result')
    plt.axis('off')
    plt.imshow(edges1, cmap=plt.cm.gray)

    # Canny edge on threshold
    edges2 = feature.canny(thresholded, 1.2)
    fig.add_subplot(2, 4, 5).title.set_text('Canny edge on threshold result')
    plt.axis('off')
    plt.imshow(edges2, cmap=plt.cm.gray)

    plt.show()

def TestSquareTracing():
    img_path = "stop_1.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(2, 3, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    stop_photo = io.imread(img_path, as_gray=True)

    fig.add_subplot(2, 3, 2).title.set_text('Grayscale result')
    plt.axis('off')
    plt.imshow(stop_photo, cmap=plt.cm.gray)

    # Threshold 
    thresholded = fltr.threshold(stop_photo, 0.82) #0.82
    fig.add_subplot(2, 3, 3).title.set_text('Threshold result')
    plt.axis('off')
    plt.imshow(thresholded, cmap=plt.cm.gray)

    # Canny edge on threshold
    edges2 = feature.canny(thresholded, 1.2)
    fig.add_subplot(2, 3, 4).title.set_text('Canny edge on threshold result')
    plt.axis('off')
    plt.imshow(edges2, cmap=plt.cm.gray)

    # Apply squaretracing algorithm to get the contour
    points = fltr.squareTracing(edges2)
    # Draw the contour 
    for point in points:
        # X and Y switched for plotting
        plt.scatter([point[1]], [point[0]], c='r', s=2)

    plt.show()
   
def TestStopSignFilter():
    # DONT WORK WELL
    #img_path = "stop_closeup.png"
    #img_path = "stop_hand.jpg"
    #img_path = "stop_w_background.jpg"

    # WORKS WELL
    #img_path = "stop_1.png"
    #img_path = "stop_sign_only.png"
    img_path = "stop_sign_pole_1.jpg"
    #img_path = "stop_sign_pole_2.png"

    fig=plt.figure(figsize=(1, 2))

    # Show original image
    img = plt.imread(img_path)
    fig.add_subplot(2, 3, 1).title.set_text('Original')
    plt.axis('off')
    plt.imshow(img) 

    # Load an image as matrix
    stop_photo = io.imread(img_path, as_gray=True)

    fig.add_subplot(2, 3, 2).title.set_text('Grayscale result')
    plt.axis('off')
    plt.imshow(stop_photo, cmap=plt.cm.gray)

    # Threshold 
    thresholded = fltr.threshold(stop_photo, 0.82) #0.82
    fig.add_subplot(2, 3, 3).title.set_text('Threshold result')
    plt.axis('off')
    plt.imshow(thresholded, cmap=plt.cm.gray)

    # Canny edge on threshold
    edges2 = feature.canny(thresholded, 1.2)
    fig.add_subplot(2, 3, 4).title.set_text('Canny edge on threshold result')
    plt.axis('off')
    plt.imshow(edges2, cmap=plt.cm.gray)

    # To print the contour on canny edge on threshold result
    fig.add_subplot(2, 3, 5).title.set_text('Contour detection result')
    plt.axis('off')
    plt.imshow(edges2, cmap=plt.cm.gray)
    
    # Get the axes of the plot
    ax = plt.gca()

    # Create the region in which we'll find contour
    s = np.linspace(0, 2*np.pi, 400)
    x = (img.shape[1]/2) + (img.shape[1]/2)*np.cos(s)
    y = (img.shape[0]/2) + (img.shape[0]/2)*np.sin(s)
    init = np.array([x, y]).T

    # Get the contour using the region we defined
    snake = active_contour(gaussian(edges2, 3), init, alpha=0.010, beta=5, gamma=0.001)

    # Plot the init region and the contour detected
    ax.plot(init[:, 0], init[:, 1], '--r', lw=1)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1)

    # Grabs the min and max coordinates of the contour to draw a rectangle
    x_min = min(snake[:,0])
    y_min = min(snake[:,1])
    x_max = max(snake[:,0])
    y_max = max(snake[:,1])
    width = x_max - x_min
    height = y_max - y_min

    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')

    # Add the patch to the Axe
    ax.add_patch(rect) 

    plt.show()

 
def main():
    
    #Test2DConvolution()

    #TestBoxFilter()

    #TestGaussianFilter()   

    #TestEdgeFilter()

    #TestGrayscaleConversion()

    #TestCannyEdgeFilter()

    #TestSquareTracing()

    TestStopSignFilter()


if __name__ == '__main__':
    main()