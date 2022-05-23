import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from common import *

from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--path", dest="dir_path", default='14-05-2022--12-38-56-865324', help="provide path to saved model")
parser.add_option("-t", "--test_set", dest="test_set", default='./dataset120k.txt', help="provide path to dataset with correct answers")

(options, args) = parser.parse_args()

dir_path = options.dir_path
test_set = options.test_set

#We create three models, each for each u
model = tf.keras.models.load_model(f'./saved_models/ANN3/{dir_path}/model')

#TODO read dataset as an input to executable
#TODO read filepath_to_model as an input to executable

# Convergence plot for u1, u2 and u3 coefficients

n=0.25

X = np.linspace(0, 0.5, 12000)
y = [np.sin(n * np.pi * x) for x in X]

predictions = model.predict(X)
predictions = predictions.flatten()

err = abs(y - predictions)

plt.title(f"Prediction, correct and error for ANN3, max err = {max(err)}")
plt.plot(X, predictions, label='prediction', color='b')
plt.plot(X, y, label='correct', color='g')
plt.plot(X, err, label='error', color='r')

plt.show()