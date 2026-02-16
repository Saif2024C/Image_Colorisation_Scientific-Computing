from PIL import Image

from generalfunctions import *
from greyscaleconversion import *
from recolorizer import *

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

recolorised_img = recolorise(D, greyscale_image)
recolorised_img.show()