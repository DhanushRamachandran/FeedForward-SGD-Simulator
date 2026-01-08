# importing all the necessary libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

import json

from activation_functions import ActivationFunction

class ActivationDerivatives(ActivationFunction):
    """
    Inherits ActivationFunction to easily access Z values if needed, 
    but primarily provides the derivative functions.
    """
    
    @staticmethod
    def relu_derivative(Z):
        """Derivative of ReLU: 1 if Z > 0, 0 otherwise."""
        return (Z > 0).astype(int)

    @staticmethod
    def sigmoid_derivative(A):
        """
        Derivative of Sigmoid: A * (1 - A)
        Requires the activation output (A), not the pre-activation (Z).
        """
        # A = sigmoid(Z)
        return A * (1 - A)

    # Note: Softmax derivative is typically handled by combining it with the 
    # Cross-Entropy loss derivative (dL/dZ = A - Y).

    def get_derivative(self, name: str):
        """Returns the derivative function reference based on the name."""
        name = name.lower()
        if name == "relu":
            return self.relu_derivative
        elif name == "sigmoid":
            return self.sigmoid_derivative
        # Add checks for other derivatives
        else:
            raise ValueError(f"Unknown activation derivative: {name}")
