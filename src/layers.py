import gzip
import pickle as pkl
import time

import grpc
import numpy as np
from sklearn.utils import shuffle

import neural_nets_pb2 as nn_pb
import neural_nets_pb2_grpc as nn_pb_grpc


class Layer(nn_pb_grpc.LayerDataExchangeServicer):
  """
  abstract layer extract common methods
  """
  def __init__(self, layer_name, upper_layer, lower_layer,
               lower_layer_nodes, current_layer_nodes,
               nonlin, nonlin_prime):
    """
    datasets : the path of mnist dataset
    nonlin: activation function
    nonlin_prime: the derivative of activation function
    """
    self.layer_name = layer_name
    self.upper_layer_addr = upper_layer
    self.lower_layer_addr = lower_layer
    self.nonlin = nonlin
    self.nonlin_prime = nonlin_prime

    # lazy initialization
    self.upper_layer_stub = None
    self.lower_layer_stub = None

    # weights dimension
    self.weights_shape = (current_layer_nodes, lower_layer_nodes)
    self.weights = None

    # record outputs from lower layer
    # use batch id as key
    # Purposes:
    # 1) used for computing the weighted sum of current layer
    # 2) used for computing the gradients for updating weights of current layer
    self.lower_layer_outputs = {}

    # computed from lower layer outputs for cache purpose
    # cache for computing delta for current layer
    # delta = partial_delta_rec * nonlin_prime(weighted_sum)
    # with different batch we have different weighted sum
    self.weighted_sum_inputs = {}


  def forward_to_upper(self, batch_id, forward_matrix, forward_labels, istrain):
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


  def backward_to_lower(self, batch_id, partial_delta, labels):
    """
    back propagate error partial_delta to lower layer
    partial_delta = dot(self.weights.T, self.delta)
    self.delta = delta_received_from_upper * nonlin_prime(z)
    """
    # create stub for lower layer
    if not self.lower_layer_stub:
      self.create_lower_stub()

    # convert partial_delta matrix to bytes string
    bytes_delta = pkl.dumps(partial_delta)
    bytes_labels = pkl.dumps(labels)

    res = self.lower_layer_stub.UpdateDelta(
      nn_pb.BackwardMsg(batch_id=batch_id,
                        delta_matrix=bytes_delta,
                        labels=bytes_labels))
    print("get response from lower layer", res.message)


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


  def init_weights(self, load_weights):
    """
    if load_weights is specified load the trained weights
    """
    if load_weights:
      # TODO 
      pass
    else:
      # x: lower layer nodes n
      # y: current layer nodes n
      x = self.weights_shape[1]
      y = self.weights_shape[0]
      self.weights = np.random.randn(y, x) / np.sqrt(x) # pylint: disable=no-member 


  # implementing rpc services
  def UpdateInput(self, request, context):
    """ Invoked by lower layer
    Once inputs updated, start computing the weighted sum
    then activation outputs,
    then forward outputs to next layer
    request: ForwardMsg
    """
    if not self.weights:
      print("Weights of {} have not initialized".format(self.layer_name))
      import sys
      sys.exit(-1)

    # get values from message
    batch_id = request.batch_id
    bytes_outputs_of_lower = request.output_matrix
    bytes_labels = request.labels
    is_train = request.is_train
    # bytes to numpy array
    outputs_of_lower = pkl.loads(bytes_outputs_of_lower)
    labels = pkl.loads(bytes_labels)

    # saving inputs
    inputs = {'matrix': outputs_of_lower,
              'labels': labels}
    self.lower_layer_outputs[batch_id] = inputs
    weighted_sum = np.dot(outputs_of_lower, self.weights)
    self.weighted_sum_inputs[batch_id] = weighted_sum

    # forward layer outputs
    activations = self.nonlin(weighted_sum) # apply element wise
    self.forward_to_upper(batch_id, activations, labels, is_train)


  def UpdateDelta(self, request, context):
    """ Invoked by upper layer
    TODO
    """





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
