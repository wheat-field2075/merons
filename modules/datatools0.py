import numpy as np
import torch
from PIL import Image

"""
Functions defined:
    - pil_to_numpy
    - numpy_to_pil
"""

""" ##############################
pil_to_numpy: read an image using pil and convert to an appropriately shaped and scaled numpy
    array. note that pil Images are [0, 255] and numpy arrays are [0, 1]

Args:
    image: the image as a pil Image
Returns:
    image: the corresponding image as a 2D numpy array
############################## """
def pil_to_numpy(image):
    image = image.convert('L') # converts from JPEGImageFile to PIL.Image, if necessary
    image = np.array(image)
    return image
""" ############################## """

""" ##############################
numpy_to_pil: given a numpy array, construct a corresponding pil Image and save it. note that pil
    Images are [0, 255] and numpy arrays are [0, 1]
    
Args:
    image: the image to be saved, presented as a numpy array
############################## """
def numpy_to_pil(image):
    image = Image.fromarray(image * 255).convert('L')
    return image
""" ############################## """