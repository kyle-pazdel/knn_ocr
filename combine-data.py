import pandas as pd
import os

file_list = os.listdir('./fonts')


def combine_data():
    df = pd.DataFrame()
    for index, file in enumerate(file_list):
        print(f"reading file: {index} {file}...")
        data = pd.read_csv(f'./fonts/{file}')
        print(f"concatenating file: {index} {file}...")
        df = pd.concat([df, data], axis=0)
        print(f"finished processing file: {index} {file}.")
    df.to_csv('fonts.csv', index=False)


def create_font_list(file_list):
    font_list = []
    for file in file_list:
        temp_tuple = os.path.splitext(file)
        font_name = temp_tuple[0]
        font_list.append(font_name)
    return font_list


# create_font_list(file_list)
# combine_data(file_list)
