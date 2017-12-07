""" output layer server"""
import time
from concurrent import futures

import grpc

import neural_nets_pb2_grpc as nn_pb2_grpc
from layers import OutputLayer
from layers_conf import output_layer as conf


def main():
  output_layer = OutputLayer(conf["layer_name"],
                             conf["lower_layer"],
                             conf["lower_layer_size"],
                             conf["num_classes"],
                             conf["learning_rate"])
  # init weights
  output_layer.init_weights(None)

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
  nn_pb2_grpc.add_LayerDataExchangeServicer_to_server(output_layer,
                                                      server)
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
