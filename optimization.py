from PIL import Image
from generalfunctions import random_bool_array, uniform_bool_array, get_D, determine_SSRI
from greyscaleconversion import RGB_to_greyscale
from recolorizer import recolorise

def optimization_function(img: Image.Image, sig1: float, sig2: float, p: float, nr_points: int):
    points = random_bool_array(img.size[0], img.size[1], nr_points)
    
    greyscale_image = RGB_to_greyscale(img)
    D = get_D(img, points)
    
    recolorised_img, computational_cost = recolorise(D, greyscale_image, sig1, sig2, p)
    
    SSRI = determine_SSRI(img, recolorised_img)
    
    print(computational_cost)
    
    return SSRI
    

homer = Image.open("Images/Homer.jpeg")
picasso2 = Image.open("Images/Picasso2.jpg")
landscape = Image.open("Images/Landscape.jpg")
baboon = Image.open("Images/baboon.png")
tulips = Image.open("Images/tulips.png")

image = homer

# points = uniform_bool_array(image.size[0], image.size[1], 30, 30)
# points = random_bool_array(image.size[0], image.size[1], 10)

sig1 = 0.2
sig2 = 18
p = 0.5

SSRI = optimization_function(image, sig1, sig2, p, 250)

print(f"Recolorized the image with an SSIM of {SSRI:.2f}.")