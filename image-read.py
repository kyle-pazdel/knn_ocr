from PIL import Image
import numpy as np


# Read and Binarize Test Image
img = np.array(Image.open("./test_images/letter_C.jpg").convert('L'))
# img = np.array(Image.open("./rgbrender_10234.png").convert('L'))
print(img)

# For Binary Black-and-White image
# The threshold value (adjust sensitivity for image binarization)
print(f"Adjusted threshold VALUE: {img.min() - 50}")
thresh = img.max() - 50
img_bool = img > thresh
inverted = np.invert(img_bool)
maxval = 255
img_bin = (inverted) * maxval
print(type(img_bin))

# # For Greyscale Image
# img_bin_keep = (inverted) * img
# print(img_bin_keep)

# Writes image to a png file
filename = "test_image"
Image.fromarray(np.uint8(img_bin)).save(f'./{filename}_read.png')

#########################################################################
