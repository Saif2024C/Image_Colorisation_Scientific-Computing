import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

def convert_png_to_array(png_file):
    '''converts a given png/jpg to an array
    requires numpy and Image from PIL
    '''
    image = Image.open(png_file) #load image
    array = np.asarray(image) #convert into array, n_pixel x n_pixels x 3 (red, green, blue)
    return array

def greyscale(array,method):
    '''
    Converts an RGB image array to greyscale using different methods. Input image of shape (n_pixels, m_pixels, 3) representing RGB values.
    method:
        - 'nor' : NTSC / perceptual weighting (0.3R + 0.59G + 0.11B)
        - 'avg' : Simple average of RGB channels
        - 'lim' : Luminance-based weighting (ITU-R BT.709)
        - 'lig' : Lightness method ((max(R,G,B) + min(R,G,B)) / 2)
    returns a 2D array of greyscale intensities. 
    '''
    greyscale_array = np.copy(array[:,:,0]) # Initialize greyscale array (copied from one channel to preserve shape/type)
    if method == 'nor': # 'nor' method: perceptual weighting based on human visual sensitivity
        greyscale_array = (array[:,:,0]*0.3               
                           +array[:,:,1]*0.59                    
                           +array[:,:,2]*0.11) 
    if method == 'avg': # 'avg' method: simple arithmetic mean of RGB channels
        greyscale_array = (array[:,:,0]+                    
                           array[:,:,1]+   
                           array[:,:,2])/3
    if method == 'lim': # 'lim' method: luminance-based conversion 
        greyscale_array = (array[:,:,0]*0.2126                       
                           +array[:,:,1]*0.7152                        
                           +array[:,:,2]*0.0722)
    if method == 'lig':  # 'lig' method: lightness conversion using HSL definition
        greyscale_array = (np.max(array[:,:,:],axis = 2)                 
                           + np.min(array[:,:,:],axis = 2))/2
    return greyscale_array #returns the array.

def uniformD(array, nx, my):
    '''
    Function takes as input the image converted to an array RGB (size: nxmx3) as well as the inputs nx and mx. Essentiall nx tells us we want every nx'th value in the x direction
    and my tells us we want every my'th value in the y direction. 
    '''
    n, m = array.shape[:2] #size of the image.
    Dpos = np.array([(xi, yi) for xi in np.arange(n)[::nx] for yi in np.arange(m)[::my]]) #we still need the positions
    rgb = array[Dpos[:,0], Dpos[:,1]]  # go into the array and get the RGB values. shape: (N, 3)
    D = np.hstack((Dpos, rgb))  # positions and RGB values into (N, 5)
    return D 

def randomD(array, N):
    '''
    Function takes as input the image converted to an array RGB (size: nxmx3 ), as well as the number of points
    N that we want to generate with colour. Then the function returns the position of those random points. It then goes into the array
    at those points to extract RGB colour, and then returns D which is a list of vecotrs of the form  (x,y, R, G, B).
    '''
    n, m = array.shape[:2] #size of the image, this way we make a random number between 0 and n and 0 and m
    Dpos = np.random.randint([0, 0], [n, m], size=(N, 2)) #random number between 0 and n and 0 and m, we want N of these and these should be tupples
    rgb = array[Dpos[:, 0], Dpos[:, 1]]  # go into the array and get the RGB values. shape: (N, 3)
    D = np.hstack((Dpos, rgb))  # positions and RGB values into (N, 5)
    return D #return shape (x,y,R,G,B) x N

def chooseD(image, N=None):
    return D

    return D
def overlay(greyscale_array,D):
    '''given the greyscale, and some information at points D this function returns an overlayed array so that one can plot the greyscaled overlayed with the colour information'''
    overlay = np.stack([greyscale_array, greyscale_array, greyscale_array], axis=2).astype(np.uint8) #axis 2 because we want this to be of shape (x,y, 3) and then we have
    #.astype(np.uint8) to ensure that the values are in between 0 and 255, strictly not needed
    for x, y, R, G, B in D: #loop over the values of D
        overlay[x, y] = [R, G, B] # Inject known color pixels
    return overlay #return the overlay
    

