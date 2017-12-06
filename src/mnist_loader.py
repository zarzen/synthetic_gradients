""" load mnist data set
convert labels to one hot representation
"""

import pickle as pkl
import numpy as np


def load_data(data_path):
  """ load mnist data, and convert labels to one hot representation"""
  with open(data_path, 'rb') as f:
    train, val, test = pkl.load(f)

  # transform labels in training data to one hot vector
  train_y = train[1]
  n = len(train_y)
  one_hot_y = np.zeros((n, 10))
  one_hot_y[np.arange(n), train_y] = 1
  train[1] = one_hot_y
  return train, val, test
