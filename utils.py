#  Created by od3ng on 20/05/2019 03:29:04 PM.
#  Project: tf-flower-classification-cloud
#  File: utils.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_folder", required=True, help="Input directory folder")
ap.add_argument("-o", "--output_folder", required=True, help="Output directory folder")
ap.add_argument("-tr", "--train", required=True, help="Sum of data training", type=float, default=0.7)
ap.add_argument("-te", "--test", required=True, help="Sum of data testing", type=float, default=0.3)
args = vars(ap.parse_args())

input = args["input_folder"]
output = args["output_folder"]
labels = []
for folder in sorted(os.listdir(input)):
    for f in os.listdir(os.path.join(input, folder)):
        labels.append(os.path.join(input, folder, f) + "," + folder)

print(len(labels))

dt = np.array(labels)

train_size = args["train"]
test_size = args["test"]

x_train, x_test = train_test_split(dt, test_size=test_size, train_size=train_size)

with open(os.path.join(output, "train_eval.txt"), 'w') as f_train_set:
    for tr in labels:
        f_train_set.write(tr + "\n")

with open(os.path.join(output, "train_set.txt"), 'w') as f_train:
    for tr in x_train:
        f_train.write(tr + "\n")

with open(os.path.join(output, "eval_set.txt"), 'w') as f_test:
    for tr in x_test:
        f_test.write(tr + "\n")
