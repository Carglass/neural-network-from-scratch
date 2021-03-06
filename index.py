# importing all the libraries and dataset
import pandas as pd
import numpy as np

from NeuralNetwork import NeuralNetwork


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, y)

for i in range(10000):
    nn.feedforward()
    nn.backprop()

print(nn.output)
