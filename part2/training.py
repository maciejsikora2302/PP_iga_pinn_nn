from cProfile import label
import os
from collections import defaultdict
from datetime import datetime
from pprint import pprint as pp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from common import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--save", dest="save", action="store_true", default=False, help="do we need to save to disk")
parser.add_option("-a", "--activation_function", dest="activation_function", default="sigmoid", help="dfefine which activation function should be used for training")
parser.add_option("-e", "--epochs", dest="epochs", default="50", help="how many epochs")
parser.add_option("-l", "--learning_rate", dest="rate", default="0.2", help="how big the learning rate")
parser.add_option("-d", "--dataset_path", dest="dataset_path", default="./dataset40k.txt", help="path to dataset")

(options, args) = parser.parse_args()

save = options.save

def larger_tanh(x):
    return 2*tf.keras.activations.tanh(x)

# LEARNING_RATE = 0.02
LEARNING_RATE = float(options.rate)
DATASET_SIZE = None
EPOCHS = int(options.epochs)

data = read_data_from_file(options.dataset_path)

# pp(data)

DATASET_SIZE = len(data['in'])

d1 = Dataset(data['in'], data['u1'])
d2 = Dataset(data['in'], data['u2'])
d3 = Dataset(data['in'], data['u3'])

datasets = [d1,d2,d3]

for d in datasets: d.normalize()
for d in datasets: d.split_datasets()

activation_function = options.activation_function
# activation_function = tf.keras.activations.get(larger_tanh)

#We create three models, each for each u
models = \
[
    tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim=1, activation=activation_function),
        tf.keras.layers.Dense(5, activation=activation_function),
        tf.keras.layers.Dense(1, activation=activation_function)
    ]) for _ in range(3)
]

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

for m in models:
    m.compile(optimizer=optimizer, loss="MSE")

#Training
for m, dataset in zip(models, datasets):
    m.fit(dataset.x_train, dataset.y_train, epochs=EPOCHS)

loss_values = []
#Testing
for i, (m, dataset) in enumerate(zip(models, datasets)):
    print(f"Evaluation for model {i+1}")
    loss_values.append(m.evaluate(dataset.x_test, dataset.y_test))

#Zapis modeli na dysk



if save:
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%d-%m-%Y--%H-%M-%S-%f")

    dir_path = f"./saved_models/{timestamp}"

    os.mkdir(dir_path)

    models[0].save(f"{dir_path}/u1")
    models[1].save(f"{dir_path}/u2")
    models[2].save(f"{dir_path}/u3")

    with open(f'{dir_path}/metadata.txt', 'w') as f:
        f.write(f"LEARNING_RATE = {LEARNING_RATE}, DATASET_SIZE = {DATASET_SIZE}, EPOCHS = {EPOCHS}\n")
        f.write(f"ACTIVATION FUNCTION: '{activation_function}'\n")
        f.write(','.join(list(map(lambda x: str(x), loss_values))))
        f.write('\n')