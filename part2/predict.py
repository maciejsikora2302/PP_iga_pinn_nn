import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from common import *

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--path", dest="dir_path", default='26-04-2022--19-03-44-837307', help="provide path to saved model")
parser.add_option("-t", "--test_set", dest="test_set", default='./dataset120k.txt', help="provide path to dataset with correct answers")

(options, args) = parser.parse_args()

# LEARNING_RATE = 0.02
LEARNING_RATE = 0.15
DATASET_SIZE = None
EPOCHS = 60

dir_path = options.dir_path
test_set = options.test_set

#We create three models, each for each u
models = \
[
    tf.keras.models.load_model(f'./saved_models/{dir_path}/u1'),
    tf.keras.models.load_model(f'./saved_models/{dir_path}/u2'),
    tf.keras.models.load_model(f'./saved_models/{dir_path}/u3')
]

#TODO read dataset as an input to executable
#TODO read filepath_to_model as an input to executable

data = read_data_from_file(test_set)

datasets = [
    Dataset(data['in'], data['u1']),
    Dataset(data['in'], data['u2']),
    Dataset(data['in'], data['u3'])
]

for d in datasets: d.normalize()

X = Dataset(data['in']).x

u_names = ['u1', 'u2', 'u3']
# Convergence plot for u1, u2 and u3 coefficients

predictions = [
    models[0].predict(X),
    models[1].predict(X),
    models[2].predict(X)
]

fig, axes = plt.subplots(3, sharex=True)

for pred, dataset, name, ax in zip(predictions, datasets, u_names, axes):
    ax.set_title(f'Prediction, correct and error for {name}')
    ax.plot(X, pred, label='prediction', color='b')
    ax.plot(X, dataset.y, label='correct', color='g')
    ax.plot(X, abs(dataset.y - pred), label='error', color='r')
    # ax.axvline(4, color='purple')
    ax.legend()

plt.show()