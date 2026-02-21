import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

from generalfunctions import *
from greyscaleconversion import *

from scipy.linalg import cho_solve, cho_factor

import time

def phi_gaussian(r: np.ndarray) -> np.ndarray:
    """Determine the values of the Gaussian RBF phi(r) = exp(-r^2) for a given array r.
    
    Args:
        r: Array of radius values.
        
    Returns:
        Array of values of the Gaussian RBF for the given array of r.
    """
    return np.exp(-r*r)

def phi_wendland(r: np.ndarray) -> np.ndarray:
    """Determine the values of the Wendland function phi(r) = (1-r)^4_+ (4r+1) for an array r, 
    here _+ indicates that we take the positive part.
    
    Args:
        r: Array of radius values.
        
    Returns:
        Array of values of the Gaussian RBF for the given array of r.
    """
    return np.power(np.maximum(1 - r, 0), 4) * (4*r + 1)

    
def kernel(
    phi, 
    x: np.ndarray, 
    y: np.ndarray, 
    greyscale_map: np.ndarray, 
    sig1 = 100, 
    sig2 = 100, 
    p = 0.5) -> np.ndarray:
    """Compute kernel matrix K of shape (m, n) where
    K_ij = phi( ||x_i - y_j||_2 / sig1 ) * phi( |g(x_i) - g(y_j)|^p / sig2 )

    Args:
        phi: Radial basis function that accepts numpy arrays (elementwise).
        x: Array-like of shape (m, 2) giving positions (row, col).
        y: Array-like of shape (n, 2) giving positions (row, col).
        greyscale_map: 2D numpy array (image) where greyscale_map[row, col] is intensity in 
            0, ..., 255.
        sig1, sig2: Positive scale factors (floats).
        p: Exponent for greyscale difference (float).

    Returns:
        K: numpy array shape (m, n).
    """
    N, M = np.shape(greyscale_map)
    input1 = cdist(x, y, metric="euclidean") / (sig1 * np.sqrt(N**2 + M**2)) 
    # cdist computes pairwise Euclidean distances, resulting in an (m, n) matrix.

    x_row = x[:, 0]
    x_col = x[:, 1]
    y_row = y[:, 0]
    y_col = y[:, 1]

    greyscale_x = greyscale_map[x_row, x_col]
    greyscale_y = greyscale_map[y_row, y_col]
    input2 = (np.abs(greyscale_x[:, None] - greyscale_y[None, :]) ** p) / (sig2 * 255**p)
    K = phi(input1) * phi(input2)
    
    return K 


def compute_as(D: np.ndarray, greyscale: np.ndarray, phi, delta=2*1e-4, **kwargs) -> np.ndarray:
    """This function computes the coefficients a_s by solving the problem (K_D + delta*n*I)a_s = f_s.
    
    Args:
        D: The positions (row, col) and known colours (r, g, b). 
        Phi: The radial basis function we want to use. 
        Delta: A constant. 
        **kwargs: To allow for arguments that we may choose or not choose to pass to the kernel 
            function, i.e. sig1, sig2 and p.
    
    Returns:
        Finds an array (n, 3) of the coefficients a_s by solving the problem above. 
    """
    KD = kernel(phi, D[:, :2], D[:, :2], greyscale, **kwargs)  #call kernel building function to build the matrix K_D. 
    
    f_s = D[:, 2:]  #extract known colours from D. Recall D = [row, col, R, G, B]
    n = len(f_s)  #the number n. i.e. the number of known positions with colour
    
    a_s = np.linalg.solve(KD + delta * n * np.eye(n), f_s) #solve 
    #notice here we are solving 3 problems, for s = r, g, and b. so a_s is of shape [n,3]
    return a_s


def get_Fs(Komega: np.ndarray, a_s: np.ndarray) -> np.ndarray:
    """Find Fs through the matrix-matrix multiplication Fs = Komega * a_s
    
    Args:
        Komega: The kernal matrix
        a_s: The coefficients that solve the problem (K_D + delta*n*I)a_s = f_s.
        
    Returns:
        The predicted colors
    
    """
    Fs =  Komega @ a_s
    return Fs

def recolorise(D: np.ndarray, 
               greyscale_image: Image.Image,
               sig1: float=300,
               sig2: float=300,
               p: float=0.5,) -> Image.Image:
    """Take the set D and the greyscale image, and use these to recolorise the image. 
    
    Args:
        D: The set of points (row, column) and their RGB values
        greyscale_image: The greyscale image input.
        
    Returns:
        The recolorised image.
    """
    greyscale_array = np.asarray(greyscale_image)[:, :, 0]
    
    phi = phi_gaussian
    kernel_params = {"sig1": sig1, "sig2": sig2, "p": p}
    a_s = compute_as(D, greyscale_array, phi, **kernel_params)
    
    start = time.time()
    
    height, width = greyscale_array.shape
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pairs = np.column_stack([rows.reshape(-1), cols.reshape(-1)])
    Komega = kernel(phi, pairs, D[:, :2], greyscale_array, **kernel_params) # compute K_omega, i.e. the kernal matrix for our whole array
    
    Fs = get_Fs(Komega, a_s).reshape(height, width, 3)
    
    end = time.time()
    total_time = end - start
    
    Fs = np.clip(Fs, 0, 255)
    
    F_img = Image.fromarray(np.uint8(Fs))
    
    return F_img, total_time
    

if __name__=="__main__":
    homer = Image.open("Images/Homer.jpeg")
    picasso2 = Image.open("Images/Picasso2.jpg")
    landscape = Image.open("Images/Landscape.jpg")
    baboon = Image.open("Images/baboon.png")
    tulips = Image.open("Images/tulips.png")
    
    image = picasso2
    width, height = image.size
    
    # points = uniform_bool_array(width, height, 10, 10)
    points = random_bool_array(width, height, 1000)
    
    greyscale_image = RGB_to_greyscale(image)
    D = get_D(image, points)

    recolorised_img, computational_cost = recolorise(D, greyscale_image)
    recolorised_img.show()
    print(f"Recolorising the image took {computational_cost:.1f} seconds")