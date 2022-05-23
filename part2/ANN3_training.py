import os
from datetime import datetime
from pprint import pprint as pp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from common import *
from optparse import OptionParser

# LEARNING_RATE = 0.02
LEARNING_RATE = 0.1
DATASET_SIZE = 1000
EPOCHS = 1200


# Collection of dataset
n = 0.25

X = np.linspace(0, 0.5 , DATASET_SIZE)

y = np.array(
    list([np.sin(n * x * np.pi)] for x in X)
)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
activation_function = 'tanh'


model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=1, activation=activation_function),
        tf.keras.layers.Dense(3, activation=activation_function),
        tf.keras.layers.Dense(1, activation=activation_function)
    ])

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss="MSE")

#Training
model.fit(x_train, y_train, epochs=EPOCHS)

save = True

if save:
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%d-%m-%Y--%H-%M-%S-%f")

    dir_path = f"./saved_models/ANN3/{timestamp}"

    os.mkdir(dir_path)

    model.save(f"{dir_path}/model")

    with open(f'{dir_path}/metadata.txt', 'w') as f:
        f.write(f"LEARNING_RATE = {LEARNING_RATE}, DATASET_SIZE = {DATASET_SIZE}, EPOCHS = {EPOCHS}\n")
        f.write(f"ACTIVATION FUNCTION: '{activation_function}'\n")
        f.write(str(model.evaluate(x_test, y_test)))
        f.write('\n')