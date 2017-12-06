""" load mnist data set
convert labels to one hot representation
"""

import pickle as pkl
import numpy as np
import gzip


def load_data(data_path):
  """ load mnist data, and convert labels to one hot representation"""
  with gzip.open(data_path, 'rb') as f:
    train, val, test = pkl.load(f, encoding='latin1')

  # transform labels in training data to one hot vector
  train_y = train[1]
  n = len(train_y)
  one_hot_y = np.zeros((n, 10))
  one_hot_y[np.arange(n), train_y] = 1
  # train[1] = one_hot_y
  return (train[0], one_hot_y), val, test
