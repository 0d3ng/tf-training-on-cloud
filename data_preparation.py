#  Created by od3ng on 22/05/2019 11:10:36 AM.
#  Project: tf-flower-classification-cloud
#  File: data_preparation.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import os
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_folder", required=True, help="Input directory folder")
ap.add_argument("-o", "--output_folder", required=True, help="Output directory folder")
ap.add_argument("-gcs", "--gcs_path", required=True, help="GCS directory folder")
ap.add_argument("-tr", "--train", type=float, default=0.8, help="Value sum of training data")
ap.add_argument("-te", "--test", type=float, default=0.2, help="Value sum of testing data")
args = vars(ap.parse_args())

FOLDERS = args["input_folder"]
OUTPUT = args["output_folder"]
BASE_GCS_PATH = args["gcs_path"]
train_size = args["train"]
test_size = args["test"]

data_array = []
data_folder = []

for folder in sorted(os.listdir(FOLDERS)):
    data_folder.append(folder)
# print(data_folder)
file_names = [os.listdir(os.path.join(FOLDERS, f)) for f in data_folder]
# print(type(file_names))
# [print(len(f)) for f in file_names]
# [print(f) for f in file_names]

file_dict = dict(zip(data_folder, file_names))
# print(type(file_dict))

for key, file_list in file_dict.items():
    # print(key, file_list)
    for file_name in file_list:
        if ".jpg" not in file_name:
            continue
        data_array.append((os.path.join(BASE_GCS_PATH, key, file_name), key))
# print(data_array)
random.shuffle(data_array)

dt = np.array(data_array)

x_train, x_test = train_test_split(dt, test_size=test_size, train_size=train_size)
# print(len(x_train))
# print(len(x_test))

with open(os.path.join(OUTPUT, "labels.txt"), 'w') as f_labels:
    for tr in data_folder:
        f_labels.write(tr + "\n")

dataframe = pd.DataFrame(x_train)
# print(dataframe)
dataframe.to_csv(os.path.join(OUTPUT, "train_set.csv"), index=False, header=False)
dataframe = pd.DataFrame(x_test)
dataframe.to_csv(os.path.join(OUTPUT, "eval_set.csv"), index=False, header=False)
dataframe = pd.DataFrame(dt)
dataframe.to_csv(os.path.join(OUTPUT, "all_data.csv"), index=False, header=False)
