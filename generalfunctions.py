import numpy as np
from PIL import Image

def get_bands(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:     
    """Split an RGB image into its Red, Green, and Blue bands.
    
    Args:
        img : Input RGB image.
        
    Returns:
        A tuple of np.ndarrays in the form (Red, Green, Blue), each channels of shape (H, W).
    """
    
    img_array = np.asarray(img)

    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]

    return red, green, blue

def random_bool_array(width: int, height: int, number_of_points: int) -> np.ndarray:
    """Generate an array (H, W, 3) consisting of random booleans, with a certain number True.
    
    Args:
        width: Width (pixels) of the image that the boolean array will be applied to.
        height: Height (pixels) of the image that the boolean array will be applied to.
        number_of_points: Number of array elements to be set to True
        
        
    Returns:
        An array (H, W, 3) of randomly picked booleans, with a certain number of array elements set 
        to True, determined by the value of number_of_points.
    """ 
    options = [True] * number_of_points + [False] * (height*width - number_of_points)
    boolean_matrix = np.random.choice(options, (height, width), replace=False)

    # Stack the boolean_matrix (H, W) to create an array of shape (H, W, 3)
    boolean_matrix_stack = np.stack((boolean_matrix, boolean_matrix, boolean_matrix), 2)

    return boolean_matrix_stack

def uniform_bool_array(width: int, height: int, w_interval: int, h_interval: int) -> np.ndarray:
    """Generate an array (H, W, 3) consisting of uniformly distributed True values.
    
    Args:
        width: Width (pixels) of the image that the boolean array will be applied to.
        height: Height (pixels) of the image that the boolean array will be applied to.
        w_interval: Number of elements set to False between each True value in the width direction.
        h_interval: Number of elements set to False between each True value in the height direction.
        
        
    Returns:
        An array (H, W, 3) of uniformly distributed True values, with a certain number of array
        elements set to False between each successive True value, determined by the interval value.
    """
    # Create base elements consisting of "interval" number of [0] elements and one [1] element 
    w_base_element = [0] * w_interval + [1]
    h_base_element = [0] * h_interval + [1]

    # We aim to fit as many base_elements into the length. The remainder, determined by 
    # trailing_zeros is filled by [0] elements.
    w_trailing_zeros = (width - 1) % (w_interval + 1)
    w_trailing_zeros_element = [0] * w_trailing_zeros
    
    h_trailing_zeros = (height - 1) % (h_interval + 1)
    h_trailing_zeros_element = [0] * h_trailing_zeros

    # Create the base_vector, containing as many base_elements as possible, and the remainder of [0] 
    # elements.
    w_base_vector = np.array(
        [1] + w_base_element * ((width - 1) // (w_interval + 1)) + w_trailing_zeros_element, 
        dtype=int
        )
    
    h_base_vector = np.array(
        [1] + h_base_element * ((height - 1) // (h_interval + 1)) + h_trailing_zeros_element, 
        dtype=int
        )
    
    base_vector_outer_product = np.outer(h_base_vector, w_base_vector)
    boolean_matrix = base_vector_outer_product.astype(bool)

    # Stack to create a boolean array of shape (H, W, 3) 
    boolean_matrix_stack = np.stack((boolean_matrix, boolean_matrix, boolean_matrix), 2)

    return boolean_matrix_stack[:height, :width, :]

def get_D(img: Image.Image, colorpoints_coords: np.ndarray) -> np.ndarray:
    """Create an array representing the set D (x, y, R, G, B) using the selected colorpoints.
    
    Args:
        img: The input RGB image.
        colorpoints_coords: A boolean array, where True indicates pixels to be left in RGB.
        
    Returns:
        An array representing the set D with columns (x, y, R, G, B)    
    """
    width, height = img.size
    red, green, blue = get_bands(img)
    
    boolean_array = colorpoints_coords[:, :, 0]
    
    row_array = np.vstack([np.ones((1,width)) * i for i in range(height)])
    col_array = np.hstack([np.ones((height,1)) * i for i in range(width)])
    
    # Determine the row, col, r, g and b for all selected points as a 1D array
    row = row_array[boolean_array].reshape(-1, 1)
    col = col_array[boolean_array].reshape(-1, 1)
    r = red[boolean_array].reshape(-1, 1)
    g = green[boolean_array].reshape(-1, 1)
    b = blue[boolean_array].reshape(-1, 1)
    
    D = np.hstack((row, col, r, g, b))
    
    return np.int64(D)
  