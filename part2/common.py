from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset():
    def __init__(self, x=None, y=None):
        if x is None: x = []
        if y is None: y = []
        self.x = np.array(x).reshape(-1,1)
        self.y = np.array(y).reshape(-1,1)
        # print(self.x)
        # print(self.y)

    def split_datasets(self, test_size = 0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size)

    def normalize(self):
        self.y = self.y / max(np.max(self.y), abs(np.min(self.y)))

def read_data_from_file(file):
    data = defaultdict(lambda: [])

    with open(file, 'r') as f:
        labels = f.readline().split(',')
        values = f.readline().split(',')
        labels = list(map(lambda x: x.split('_')[0], labels))
        for l, v in zip(labels, values):
            data[l].append(float(v))
    
    return data