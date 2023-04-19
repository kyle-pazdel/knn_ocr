from PIL import Image
import numpy as np
import csv

test_number = 40

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


data_array = convert_data()
parsed_data = parse_colors(data_array)
image_array = reshape_array(parsed_data[test_number])


# Writes image to a png file
def write_image(image_array):
    width = 20
    height = 20
    channels = 3

    array = np.zeros([height, width, channels], dtype=np.uint8)
    for index, value in enumerate(image_array):
        for j_index, j in enumerate(image_array[index]):
            array[index, j_index] = [j, j, j]
    img = Image.fromarray(array)
    img.save(f'rgbrender_{test_number}.png')


write_image(image_array)
