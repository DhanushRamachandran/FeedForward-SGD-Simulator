
# defining the componenents
# importing all the necessary libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

import json

class ActivationFunction:
    # Static methods are used because the activation function doesn't depend 
    # on the state of the ActivationFunction class instance.
    
    @staticmethod
    def relu(Z):
        """ReLU Activation: f(Z) = max(0, Z)"""
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid(Z):
        """Sigmoid Activation: f(Z) = 1 / (1 + exp(-Z))"""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def softmax(Z):
        """Softmax Activation for output layer probabilities."""
        # Ensure numerical stability
        exp_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def get_activation(self, name: str):
        """Returns the function reference based on the name."""
        name = name.lower()
        if name == "relu":
            return self.relu
        elif name == "sigmoid":
            return self.sigmoid
        elif name == "softmax":
            return self.softmax
        else:
            raise ValueError(f"Unknown activation function: {name}")