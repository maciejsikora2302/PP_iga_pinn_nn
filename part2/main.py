from collections import defaultdict
from pprint import pprint as pp
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.15
DATASET_SIZE = None
EPOCHS = 100

class Dataset():
    def __init__(self, x=None, y=None):
        if x is None: x = []
        if y is None: y = []
        self.x = np.array(x).reshape(-1,1)
        self.y = np.array(y).reshape(-1,1)

    def split_datasets(self, test_size = 0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size)



data = defaultdict(lambda: [])

with open('./dataset40k.txt', 'r') as f:
    labels = f.readline().split(',')
    values = f.readline().split(',')
    labels = list(map(lambda x: x.split('_')[0], labels))
    for l, v in zip(labels, values):
        data[l].append(float(v))

# pp(data)

DATASET_SIZE = len(data['in'])

d1 = Dataset(data['in'], data['u1'])
d2 = Dataset(data['in'], data['u2'])
d3 = Dataset(data['in'], data['u3'])

datasets = [d1,d2,d3]

for d in datasets: d.split_datasets()

#We create three models, each for each u
models = \
[
    tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=1, activation='sigmoid'),
        tf.keras.layers.Dense(3, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
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

with open('results.txt', 'a') as f:
    f.write(f"LEARNING_RATE = {LEARNING_RATE}, DATASET_SIZE = {DATASET_SIZE}, EPOCHS = {EPOCHS}\n")
    f.write(','.join(list(map(lambda x: str(x), loss_values))))
    f.write('\n')

