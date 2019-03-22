# importing all the libraries and dataset
import pandas as pd
import numpy as np

from NeuralNetwork import NeuralNetwork

# Package imports
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
# SciKitLearn is a machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


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
