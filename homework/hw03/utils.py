import os
import numpy as np
from PIL import Image

def save_array_as_image(array, filename, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Normalize the array to 0-255
    array = ((array - np.min(array)) / (np.max(array) - np.min(array))) * 255
    array = array.astype(np.uint8)

    # Save the array as an image
    img = Image.fromarray(array)
    img.save(os.path.join(directory, filename))