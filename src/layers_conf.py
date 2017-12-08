""" contains configuration objects for each layer"""


input_layer = {
  "layer_name": "input",
  "listen_on": "[::]:50050",
  "upper_layer": "127.0.0.1:50051",
  "input_dim": 784,
  "data_path": "../data/mnist.pkl.gz",
  "batch_size": 100,
  "epochs": 5
}

hidden_layer1 = {
  "layer_name":"hidden1",
  "listen_on": "[::]:50051",
  "lower_layer" : "127.0.0.1:50050",
  "upper_layer" : "127.0.0.1:50052",
  "lower_layer_size": 784,
  "layer_size": 500,
  "learning_rate": 0.001,
  "enable_SG": False,
  "sg_learning_rate": 0.0001
}

output_layer = {
  "layer_name": "output",
  "listen_on": "[::]:50052",
  "lower_layer": "127.0.0.1:50051",
  "lower_layer_size": 500,
  "num_classes": 10,
  "learning_rate": 0.001
}
