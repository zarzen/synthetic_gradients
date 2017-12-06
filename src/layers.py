import gzip
import pickle as pkl
import time
from datetime import datetime

import grpc
import numpy as np
from sklearn.utils import shuffle

import neural_nets_pb2 as nn_pb
import neural_nets_pb2_grpc as nn_pb_grpc
from mnist_loader import load_data
from activations import *


# pylint: disable=too-many-arguments


class Layer(nn_pb_grpc.LayerDataExchangeServicer):
  """
  abstract layer extract common methods
  """
  # pylint: disable=too-many-arguments
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


  def update_weights(self, lr, delta, outputs_of_lower):
    """
    outputs of lower: equals to inputs of this layer
    """
    delta_shape = delta.shape
    inputs_shape = outputs_of_lower.shape
    delta = delta.reshape(delta_shape[0], delta_shape[1], 1)
    inputs = outputs_of_lower.reshape(inputs_shape[0], 1, inputs_shape[1])

    gradients = delta * inputs
    gradients_avg = np.mean(gradients, axis=0)

    self.weights = self.weights - lr * gradients_avg


  # implementing rpc services
  def UpdateInput(self, request, context):
    # implemented in Hidden Layer and Output Layer
    pass


  def UpdateDelta(self, request, context):
    """ Invoked by upper layer
    will be implemented by hidden layer
    """
    pass



class InputLayer(Layer):
  """ for input data"""

  def __init__(self, upper_layer, data_path, input_dim, layer_name="input"):
    super().__init__(layer_name, upper_layer,
                     None, None, input_dim,
                     None, None)

    self.train, self.val, self.test = load_data(data_path)

  def start_feed_data(self, batch_size, epochs):
    """"""
    #for i in range(epochs):


  def UpdateInput(self, req, ctx):
    """"""
    print("Should not have lower layer")

  def UpdateDelta(self, req, ctx):
    """"""
    batch_id = req.batch_id
    print("Complete backpropagation for batch {} at {}".format(
      batch_id,
      datetime.now().strftime("%Y-%m-%d %H:%M:%S")))



class HiddenLayer(Layer):
  """ hidden layer"""

  def __init__(self, layer_name,
               upper_layer,
               lower_layer,
               lower_layer_size,
               layer_size,
               nonlin,
               nonlin_prime,
               learning_rate,
               enable_synthetic_gradients
               ):
    """
    enable_synthetic_gradients: whether use synthetic gradients
      to do error approximating
    """
    super().__init__(layer_name, upper_layer,
                     lower_layer, lower_layer_size,
                     layer_size, nonlin,
                     nonlin_prime)
    self.lr = learning_rate
    self.enable_sg = enable_synthetic_gradients
    self.sg_weights = None


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

    # update weights immediately with SG, if enabled SG
    # TODO


  def UpdateDelta(self, req, ctx):
    """
    delta shape: (batch_size, size_of_current_layer)
    req: BackwardMsg
    """
    batch_id = req.batch_id
    bytes_partial_delta = req.partial_delta
    partial_delta = pkl.loads(bytes_partial_delta)
    bytes_labels = req.labels # variable currently not useful
    labels = pkl.loads(bytes_labels)

    # compute delta for current layer
    z = self.weighted_sum_inputs[batch_id]
    z_nonlin_prime = self.nonlin_prime(z)

    # shape of delta: (batch_size, size_of_layer)
    delta = partial_delta * z_nonlin_prime

    # compute partial delta for lower layer
    partial_delta_for_lower = np.dot(delta, self.weights.transpose())
    # send partial delta to lower layer
    self.backward_to_lower(batch_id,
                           partial_delta_for_lower,
                           labels)

    if self.enable_sg:
      # TODO train the SG
      pass
    else:
      # update weights regularly
      inputs = self.lower_layer_outputs[batch_id]
      self.update_weights(self.lr, delta, inputs)


    # delete stored for weighted sum
    del self.weighted_sum_inputs[batch_id]
    # delete stored for lower layer outputs
    del self.lower_layer_outputs[batch_id]



class OutputLayer(Layer):
  """ output layer
  computing the error based on labels and prediction
  using softmax as output activations and cross entropy loss

  """
  def __init__(self, lower_layer, lower_layer_size, num_classes, learning_rate ):
    super().__init__("output", None,
                     lower_layer,
                     lower_layer_size,
                     num_classes,
                     None,
                     None)
    self.lr = learning_rate

  def UpdateInput(self, req, ctx):
    """ once received input from lower layer:
    compute weighted sum -> softmax output -> loss -> back propagate
    """
    batch_id = req.batch_id
    bytes_outputs_of_lower = req.output_matrix
    bytes_labels = req.labels
    is_train = req.is_train

    outputs_of_lower = pkl.loads(bytes_outputs_of_lower)
    labels = pkl.loads(bytes_labels)

    weighted_sum = np.dot(outputs_of_lower, self.weights)
    softmax_output = softmax(weighted_sum, axis=1)
    delta = softmax_output - labels

    if is_train:
      # compute delta for lower layer first
      # because current error is based on current weights
      partial_delta_for_lower = np.dot(delta, self.weights.transpose())
      # send to lower layer
      self.backward_to_lower(batch_id, partial_delta_for_lower, labels)

    # update weights
    self.update_weights(self.lr, delta, outputs_of_lower)


  def UpdateDelta(self, req, ctx):
    """ No upper layer"""
    print("Error: No upper layer for output layer")
