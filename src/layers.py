import gzip
import pickle as pkl
import time

import grpc
import numpy as np
from sklearn.utils import shuffle

import neural_nets_pb2 as nn_pb
import neural_nets_pb2_grpc as nn_pb_grpc


class Layer():
  """
  abstract layer extract common methods
  """
  def __init__(self, upper_layer, lower_layer):
    """
    datasets : the path of mnist dataset
    """
    self.upper_layer_addr = upper_layer
    self.lower_layer_addr = lower_layer

    # lazy initialization
    self.upper_layer_stub = None
    self.lower_layer_stub = None

  def forward_upper(self, batch_id, forward_matrix, forward_labels, istrain):
    """
    forward output to upper layer
    """
    if not self.upper_layer_stub:
      self.create_upper_stub()

    # convert numpy array to byte string
    bytes_matrix = pkl.dumps(forward_matrix, 2)
    bytes_labels = pkl.dumps(forward_labels, 2)

    # send message to next layer
    res = self.upper_layer_stub.UpdateInput(
      nn_pb.ForwardMsg(batch_id=batch_id,
                       output_matrix=bytes_matrix,
                       labels=bytes_labels,
                       is_train=istrain))
    print("get response form upper layer", res.message)

  def backward_lower(self, batch_id, delta, labels):
    """
    back propagate error delta to lower layer
    """
    # TODO

  def create_upper_stub(self):
    """ create upper_layer_stub for exchanging data between grpc"""
    if self.upper_layer_addr:
      channel = grpc.insecure_channel(self.upper_layer_addr)
      self.upper_layer_stub = nn_pb_grpc.LayerDataExchangeStub(channel)
    else:
      print("no upper layer has been specified")


  def create_lower_stub(self):
    """ stub for lower layer communication"""
    if self.lower_layer_addr:
      channel = grpc.insecure_channel(self.lower_layer_addr)
      self.lower_layer_stub = nn_pb_grpc.LayerDataExchangeStub(channel)
    else:
      print("no lower layer has been specified")



class InputLayer(Layer):
  """ for input data"""

  def __init__(self, upper_layer, data_path):
    super().__init__(upper_layer, None)

    self.train, self.val, self.test = self.load_data(data_path)

  def start_train(self, batch_size, epochs):
    """"""
    #for i in range(epochs):

  def load_data(self, data_path):
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
