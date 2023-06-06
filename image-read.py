from PIL import Image
import numpy as np
import cv2


# Read and Binarize Test Image
img = np.array(Image.open("./test_images/letter_C.jpg").convert('L'))
img = np.array(Image.open("./rgbrender_10234.png").convert('L'))
img = cv2.resize(img, dsize=(20, 20), interpolation=cv2.INTER_LINEAR)
img = img.flatten()
# img = cv2.resize(img, dsize=(409, 1))
img = [img]
img = np.array(img)
print(img)
print(img.shape)


# For Binary Black-and-White image
# The threshold value (adjust sensitivity for image binarization)
print(f"Adjusted threshold VALUE: {img.min() - 50}")
thresh = img.max() - 50
img_bool = img > thresh
inverted = np.invert(img_bool)
maxval = 255
img_bin = (inverted) * maxval
print(f"img_bin:   {img_bin}")
print(type(img_bin))

# # For Greyscale Image
# img_bin_keep = (inverted) * img
# print(img_bin_keep)

# Writes image to a png file
filename = "test_image"
Image.fromarray(np.uint8(img_bin)).save(f'./{filename}_read.png')

#########################################################################
