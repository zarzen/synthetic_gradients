""" hidden layer 1"""
import time
from concurrent import futures

import grpc

import neural_nets_pb2_grpc as nn_pb2_grpc
from activations import *
from layers import HiddenLayer
from layers_conf import hidden_layer1 as conf


def main():
  """"""
  hidden_layer = HiddenLayer(conf["layer_name"],
                             conf["upper_layer"],
                             conf["lower_layer"],
                             conf["lower_layer_size"],
                             conf["layer_size"],
                             relu,
                             relu_prime,
                             conf["learning_rate"],
                             conf["enable_SG"])
  # weights initialization
  hidden_layer.init_weights(None)

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
  nn_pb2_grpc.add_LayerDataExchangeServicer_to_server(hidden_layer,
                                                      server)
  # listen on
  server.add_insecure_port(conf["listen_on"])
  server.start()

  # idle
  try:
    while True:
      time.sleep(24*60*60)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  main()
