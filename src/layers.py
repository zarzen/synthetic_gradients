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
    self.biases = None

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
    # print("get response form upper layer", res.message)


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
                        partial_delta=bytes_delta,
                        labels=bytes_labels))
    # print("get response from lower layer", res.message)


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


  def init_weights(self, load_weights=None):
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
      self.biases = np.random.randn(y, 1) # pylint: disable=no-member


  def check_weights(self):
    if self.weights is None or self.biases is None:
      print("Weights of {} have not initialized".format(self.layer_name))
      import sys
      sys.exit(-1)


  def update_weights(self, lr, delta, outputs_of_lower):
    """
    outputs of lower: equals to inputs of this layer
    """
    delta_shape = delta.shape
    inputs_shape = outputs_of_lower.shape

    # update biases
    avg_delta = np.mean(delta, axis=0).reshape(delta_shape[1], 1)
    self.biases = self.biases - lr * avg_delta

    # compute gradients for weights
    delta = delta.reshape(delta_shape[0], delta_shape[1], 1)
    inputs = outputs_of_lower.reshape(inputs_shape[0], 1, inputs_shape[1])
    gradients = delta * inputs
    gradients_avg = np.mean(gradients, axis=0)

    self.weights = self.weights - lr * gradients_avg


  def parse_forward_msg(self, req):
    """ extract and transform data in forward message"""
    batch_id = req.batch_id
    bytes_outputs_of_lower = req.output_matrix
    bytes_labels = req.labels
    is_train = req.is_train

    outputs_of_lower = pkl.loads(bytes_outputs_of_lower)
    labels = pkl.loads(bytes_labels)
    return batch_id, outputs_of_lower, labels, is_train


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
    train_X = self.train[0]
    train_y = self.train[1]
    val_X = self.val[0]
    val_y = self.val[1]
    train_size = train_X.shape[0]
    batch_id = 0
    test_batch_id = -1 # use negative number, diff with batch_id
    for i in range(epochs):
      print("Start feed {0} epoch data".format(i))

      train_X, train_y = shuffle(train_X, train_y)
      for j in range(0, train_size, batch_size):
        minibatch_X = train_X[j:j+batch_size]
        minibatch_y = train_y[j:j+batch_size]
        self.forward_to_upper(batch_id, minibatch_X, minibatch_y, True)
        batch_id += 1

      # send test data for evaluation
      self.forward_to_upper(test_batch_id, val_X, val_y, False)
      test_batch_id -= 1


  def UpdateInput(self, req, ctx):
    """"""
    print("Should not have lower layer")
    return nn_pb.PlainResponse(message="Wrong invoke!")


  def UpdateDelta(self, req, ctx):
    """"""
    batch_id = req.batch_id
    if batch_id % 100 == 0:
      print("Complete backpropagation for batch {} at {}".format(
        batch_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    return nn_pb.PlainResponse(message="Received at layer {}".format(
      self.layer_name))



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
               enable_synthetic_gradients,
               sg_learning_rate
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
    self.sg_lr = sg_learning_rate
    self.sg_weights = None
    self.sg_deltas = {}


  def init_sg_weights(self):
    """ using linear synthetic gradients model
    SG(h, y) = hA + yB + C
    refer to paper, Understanding synthetic gradients and decoupled neural networks
    """
    n = self.weights_shape[0] # size of current layer
    # pylint: disable=no-member
    A = np.random.randn(n, n) / np.sqrt(n)
    B = np.random.randn(10, n) / np.sqrt(n)
    C = np.random.randn(1, n) / np.sqrt(n)
    # pylint: enable=no-member
    self.sg_weights = [A, B, C]

  def check_sg_weights(self):
    if self.sg_weights is None:
      self.init_sg_weights()


  def SG(self, h, y):
    """ generate delta by weighted sum and label

    h: outputs of this layer
    y: labels for this batch
    """
    self.check_sg_weights()

    A = self.sg_weights[0] #(n, n)
    B = self.sg_weights[1] #(10, n)
    C = self.sg_weights[2] #(1, n)

    delta = np.matmul(h, A) + np.matmul(y, B) + C
    return delta

  def update_sg_weights(self, true_delta, batch_id):
    """ name conventions refer paper :
       Understanding synthetic gradients and decoupled neural interface
       TODO: synthetic gradient estimates the partial delta instead true gradients
    """
    sg_delta = self.sg_deltas[batch_id]
    weighted_sum = self.weighted_sum_inputs[batch_id]
    labels = self.lower_layer_outputs[batch_id]['labels']
    y = labels

    h = self.nonlin(weighted_sum)

    Err = sg_delta - true_delta
    A = self.sg_weights[0] - self.sg_lr * 2 * np.dot(h.transpose(), Err) / h.shape[0]
    B = self.sg_weights[1] - self.sg_lr * 2 * np.dot(y.transpose(), Err) / y.shape[0]
    C = self.sg_weights[2] - self.sg_lr * 2 * np.mean(Err, axis=0)

    self.sg_weights = [A, B, C]

    # del stored delta
    del self.sg_deltas[batch_id]


  def UpdateInput(self, request, context):
    """ Invoked by lower layer
    Once inputs updated, start computing the weighted sum
    then activation outputs,
    then forward outputs to next layer
    request: ForwardMsg
    """
    self.check_weights()

    # get values from message
    batch_id, outputs_of_lower, labels, is_train = self.parse_forward_msg(request)
    print("Get inputs id: {0}, matrix shape: {1}, labels shape: {2}".format(
      batch_id, outputs_of_lower.shape, labels.shape))

    weighted_sum = np.dot(outputs_of_lower, self.weights.transpose()) \
                   + self.biases.transpose()
    # saving inputs during training, because for weights updating
    if is_train:
      inputs = {'matrix': outputs_of_lower,
                'labels': labels}
      self.lower_layer_outputs[batch_id] = inputs
      self.weighted_sum_inputs[batch_id] = weighted_sum

    activations = self.nonlin(weighted_sum) # apply element wise

    # update weights immediately with SG, if enabled SG
    if self.enable_sg and is_train:
      print("update weights based on SG delta")
      sg_delta = self.SG(activations, labels)
      # TODO use sg_delta to compute the gradients by sg_delta * self.nonline_prime(z)
      self.update_weights(self.lr, sg_delta, outputs_of_lower)
      self.sg_deltas[batch_id] = sg_delta

    # forward layer outputs
    self.forward_to_upper(batch_id, activations, labels, is_train)
    print("batch id: {0}, activations shape {1}".format(
      batch_id, activations.shape))

    # return received
    return nn_pb.PlainResponse(message="Inputs received by layer {}".format(
      self.layer_name))


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
    partial_delta_for_lower = np.dot(delta, self.weights)
    # send partial delta to lower layer
    self.backward_to_lower(batch_id,
                           partial_delta_for_lower,
                           labels)

    if self.enable_sg:
      # train the SG
      # TODO pass partial delta instead
      self.update_sg_weights(delta, batch_id)
    else:
      # update weights regularly
      inputs = self.lower_layer_outputs[batch_id]['matrix']
      self.update_weights(self.lr, delta, inputs)


    # delete stored for weighted sum
    del self.weighted_sum_inputs[batch_id]
    # delete stored for lower layer outputs
    del self.lower_layer_outputs[batch_id]

    return nn_pb.PlainResponse(
      message="Partial delta received at {}".format(self.layer_name))



class OutputLayer(Layer):
  """ output layer
  computing the error based on labels and prediction
  using softmax as output activations and cross entropy loss

  """
  def __init__(self, layer_name, lower_layer, lower_layer_size,
               num_classes, learning_rate ):
    super().__init__(layer_name, None,
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
    self.check_weights()

    batch_id, outputs_of_lower, labels, is_train = self.parse_forward_msg(req)

    weighted_sum = np.dot(outputs_of_lower, self.weights.transpose()) \
                   + self.biases.transpose()
    softmax_output = softmax(weighted_sum, axis=1)
    # print("weighted sum", weighted_sum)
    # print("outputs of lower", outputs_of_lower)

    if is_train:
      delta = softmax_output - labels
      # compute delta for lower layer first
      # because current error is based on current weights
      partial_delta_for_lower = np.dot(delta, self.weights)
      # send to lower layer
      self.backward_to_lower(batch_id, partial_delta_for_lower, labels)

      # cross entropy loss
      if batch_id % 100 == 0:
        total_loss = np.log(softmax_output) * labels # pylint: disable=no-member
        # print("total loss: ", np.sum(total_loss))
        loss = -1 * np.sum(total_loss) / labels.shape[0]
        print("For batch id {}, avg loss: {}".format(batch_id, loss))

      # update weights
      self.update_weights(self.lr, delta, outputs_of_lower)

    else:
      # test evaluation
      pred_results = np.argmax(softmax_output, axis=1)
      matched = sum(int(y == t) for (y, t) in zip(pred_results, labels))
      print("Epoch {}, Performance test {} / {}".format(
        -1*batch_id, matched, labels.shape[0]))


    return nn_pb.PlainResponse(message="Inputs received at {}".format(
      self.layer_name))


  def UpdateDelta(self, req, ctx):
    """ No upper layer"""
    print("Error: No upper layer for output layer")
    return nn_pb.PlainResponse(message="Invalid Operation!!")
