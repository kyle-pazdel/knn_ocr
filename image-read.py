from PIL import Image
import numpy as np

img = np.array(Image.open("./test_images/letter_C.jpg").convert('L'))
# img = np.array(Image.open("./rgbrender_10234.png").convert('L'))
print(img)
thresh = 200
img_bool = img > thresh
print(img_bool)
maxval = 255
img_bin = (img > thresh) * maxval
print(img_bin)
img_bin_keep = (img > thresh) * img
print(img_bin_keep)

filename = "test_image"
Image.fromarray(np.uint8(img_bin_keep)).save(f'./{filename}_read.png')

# # Reads file and converts to numpy array
# filename = "rgbrender_13456"


# img = Image.open(f"./{filename}.png")

##########################################################
# numpydata = np.asarray(img)

# print(numpydata)

# # Writes image to a png file


# def write_image(image_array):
#     img = Image.fromarray(image_array)
#     img.save(f'{filename}_read.png')


# write_image(numpydata)
