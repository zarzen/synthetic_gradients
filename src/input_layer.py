""" Input Layer"""
import time
from concurrent import futures

import grpc

import neural_nets_pb2_grpc as nn_pb2_grpc
from layers import InputLayer
from layers_conf import input_layer as conf


def main():
  """"""
  input_layer = InputLayer(conf["upper_layer"],
                           conf["data_path"],
                           conf["input_dim"],
                           conf["layer_name"])

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
  nn_pb2_grpc.add_LayerDataExchangeServicer_to_server(input_layer,
                                                      server)
  server.add_insecure_port(conf["listen_on"])
  server.start()

  input("Press Enter to start sending data...")

  input_layer.start_feed_data(conf["batch_size"], conf["epochs"])

  # idle
  try:
    while True:
      time.sleep(24*60*60)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  main()
