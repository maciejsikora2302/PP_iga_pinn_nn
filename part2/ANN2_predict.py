from random import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from common import *

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--path", dest="dir_path", default='23-05-2022--18-40-03-293135', help="provide path to saved model")
parser.add_option("-t", "--test_set", dest="test_set", default='./dataset120k.txt', help="provide path to dataset with correct answers")

(options, args) = parser.parse_args()

# LEARNING_RATE = 0.02
DATASET_SIZE = 1000

dir_path = options.dir_path
test_set = options.test_set

#We create three models, each for each u
model = tf.keras.models.load_model(f'./saved_models/ANN2/{dir_path}/model')


n=0.25

X = np.linspace(0, 0.5, DATASET_SIZE)


predictions = model.predict([(random(), x) for x in X])
predictions = predictions.flatten()

Y = [np.sin(math.pi * x * n) for x in X]

err = abs(predictions - Y)
    

plt.title(f"Prediction, correct and error for ANN3, max err = {max(err)}")
plt.plot(X, predictions, label='prediction', color='b')
plt.plot(X, Y, label='correct', color='g')
plt.plot(X, err, label='error', color='r')

plt.show()