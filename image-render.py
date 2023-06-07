from PIL import Image
import numpy as np
import csv

test_number = 13456

# Converts data from csv to a two-dimensional array
# MUST BE greater than 0 and less than 23270
# This range is the entire ARIAL Dataset


def convert_data():
    results = []
    with open("../ARIAL.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            results.append(row)
    return results


# Parses all RGB related data from array
def parse_colors(array):
    res = []
    for row in array:
        new_row = []
        for col in row[12:]:
            new_row.append(col)
        res.append(new_row)
    return res


# Reshapes_array to two-dimensional to be iteratable for write method
def reshape_array(array):
    formatted = np.array(array)
    reshaped = formatted.reshape(20, 20)
    print(reshaped)
    return reshaped

# Binarize Dataset


def binarize_train_images(img):
    # The threshold value (adjust sensitivity for image binarization)
    print(f"Adjusted threshold VALUE: {img.min() - 50}")
    thresh = img.max() - 50
    img_bool = img > thresh
    inverted = np.invert(img_bool)
    maxval = 255
    img_bin = (inverted) * maxval
    # print("BIN IMAGE: ", img_bin)
    # print("BIN IMG ELEMENT TYPE:   ----", type(img_bin[0][0]))
    return img_bin


data_array = convert_data()
parsed_data = parse_colors(data_array)
image_array = reshape_array(parsed_data[test_number])
image_array = np.array(image_array)
print("ARRAY::: ", image_array)
binarized_image = binarize_train_images(image_array)
print("BIN ARRAY::: ", binarized_image)

# Writes image to a png file
Image.fromarray(np.uint8(image_array)).save(f'./rgbrender_{test_number}_2.png')
# def write_image(image_array):
#     width = 20
#     height = 20
#     channels = 3

#     array = np.zeros([height, width, channels], dtype=np.uint8)
#     for index, value in enumerate(image_array):
#         for j_index, j in enumerate(image_array[index]):
#             array[index, j_index] = [j, j, j]
#     img = Image.fromarray(array)
#     img.save(f'rgbrender_{test_number}.png')


# write_image(image_array)
