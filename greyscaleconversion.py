import numpy as np
from PIL import Image
from generalfunctions import *

def convert_bands_to_greyscale(
    red: np.ndarray, 
    green: np.ndarray, 
    blue: np.ndarray, 
    method: str) -> np.ndarray:
    """Convert RGB bands into an array of greyscale values.
    
    Args:
        red: Array (H, W) for the red band.
        blue: Array (H, W) for the blue band.
        green: Array (H, W) for the green band.
        method: RGB to greyscale conversion method, with the following options:
            - "nor": perceptual weighting based on human visual sensitivity
            - "avg": arithmetic mean of RGB channels
            - "lum": luminance-based conversion
            - "lig": lightness conversion using HSL definition
        
    Returns:
        An array of shape (H, W, 3) representing the greyscale image.
    """
    if method == "nor":
        grey_float = 0.3 * red + 0.59 * green + 0.11 * blue
    elif method == "avg":
        grey_float = (red + green + blue) / 3
    elif method == "lum":
        grey_float = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    elif method == "lig":
        grey_float = (np.maximum.reduce([red, blue, green]) \
                      + np.minimum.reduce([red, blue, green])) / 2    

    # Convert the resulting array of floats (H, W) into 8-bit.
    grey_int = np.uint8(grey_float)

    # Duplicate the GreyInt array (H, W) across all bands to get (H, W, 3).
    grey = np.stack((grey_int, grey_int, grey_int), 2)

    return grey

def RGB_to_greyscale(img: Image.Image) -> Image.Image:
    """Convert RGB image to a greyscale image.
    
    Args:
        img: Input RGB image.
        
    Returns:
        Greyscale version of the input image.
    """
    red, green, blue = get_bands(img)

    grey = convert_bands_to_greyscale(red, green, blue, "nor")

    greyscale_img = Image.fromarray(grey)

    return greyscale_img

def generate_mixed_img(img: Image.Image, colorpoints_coords: np.ndarray) -> Image.Image:
    """Create a mixed color and greyscale image, with the position of colorpoints predetermined.  
    
    Args:
        img: The input RGB image.
        colorpoints_coords: A boolean array, where True indicates pixels to be left in RGB.
        
    Returns:
        A mixed color and greyscale image.
        
    """
    color_img = img
    greyscale_img = RGB_to_greyscale(img)

    color_img_array = np.asarray(color_img).copy()
    greyscale_img_array = np.asarray(greyscale_img).copy()

    mixed_img_array = np.where(colorpoints_coords, color_img_array, greyscale_img_array)

    return Image.fromarray(mixed_img_array)

if __name__ == "__main__":
    homer = Image.open("Images/Homer.jpeg")
    picasso2 = Image.open("Images/Picasso2.jpg")
    landscape = Image.open("Images/Landscape.jpg")
    baboon = Image.open("Images/baboon.png")
    tulips = Image.open("Images/tulips.png")
    
    image = tulips
    width, height = image.size
    
    # points = uniform_bool_array(width, height, 2, 2)
    points = random_bool_array(width, height, 100)

    mixed_image = generate_mixed_img(image, points)
    mixed_image.show()