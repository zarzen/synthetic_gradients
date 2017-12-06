""" contains activation functions and its corresponding derivative"""

import numpy as np



# pylint: disable=no-member

def relu(weighted_sum):
  return np.maximum(0, weighted_sum)


def relu_prime(weighted_sum):
  return np.minimum(1, np.maximum(0, weighted_sum))


def softmax(weighted_sum, axis=0):
  exp_z = np.exp(weighted_sum)
  return exp_z / np.sum(exp_z, axis).reshape(-1, 1)
