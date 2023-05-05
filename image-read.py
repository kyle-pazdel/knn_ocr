from PIL import Image
import numpy as np
# from numpy import asarray
filename = "rgbrender_13456"

img = Image.open(f"./{filename}.png")
numpydata = np.asarray(img)

print(numpydata)

# Writes image to a png file


def write_image(image_array):
    img = Image.fromarray(image_array)
    img.save(f'{filename}_read.png')


write_image(numpydata)
