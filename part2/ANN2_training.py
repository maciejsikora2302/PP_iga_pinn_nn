from cProfile import label
import os
from collections import defaultdict
from datetime import datetime
from pprint import pprint as pp
import numpy as np
from pyparsing import col
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from common import *
from optparse import OptionParser
import math
import pandas as pd

# LEARNING_RATE = 0.02
LEARNING_RATE = 0.1
DATASET_SIZE = 500
EPOCHS = 100



# Collection of dataset
N = np.linspace(0, 0.5, DATASET_SIZE)
X = np.linspace(0, 0.5, DATASET_SIZE)

Y = [[np.array([np.array([n,x]), np.sin(math.pi * x * n)]) for x in X] for n in N]

data = []
value = []

for item in Y:
    for item2 in item:
        data.append(item2[0])
        value.append(item2[1])

data = np.array(data)
value = np.array(value)

print(data[0])
print(value[0])

# y = np.sin(math.pi * x * n)
# data = {'n' : n, 'x' : x}
# df = pd.DataFrame(data)

x_train, x_test, y_train, y_test = train_test_split(data, value, test_size = 0.2)
activation_function = 'tanh'

# Create model, input_dim = 2 because we have n and x as an input
model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim=2, activation=activation_function),
        tf.keras.layers.Dense(5, activation=activation_function),
        tf.keras.layers.Dense(1, activation=activation_function)
    ])

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="MSE")
model.fit(x_train, y_train, epochs=EPOCHS)

save = True

if save:
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%d-%m-%Y--%H-%M-%S-%f")

    dir_path = f"./saved_models/ANN2/{timestamp}"

    os.mkdir(dir_path)

    model.save(f"{dir_path}/model")

    with open(f'{dir_path}/metadata.txt', 'w') as f:
        f.write(f"LEARNING_RATE = {LEARNING_RATE}, DATASET_SIZE = {DATASET_SIZE}, EPOCHS = {EPOCHS}\n")
        f.write(f"ACTIVATION FUNCTION: '{activation_function}'\n")
        f.write(str(model.evaluate(x_test, y_test)))
        f.write('\n')




