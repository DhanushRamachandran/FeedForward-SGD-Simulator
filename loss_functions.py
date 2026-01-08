# importing all the necessary libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

import json
class LossFunction:
    
    @staticmethod
    def cross_entropy(y_true_onehot, y_pred_probs):
        """Calculates Categorical Cross-Entropy Loss (for classification)."""

        y_true_onehot = y_true_onehot.T
        y_pred_probs = y_pred_probs.T

        print("y_true_onehot: ",y_true_onehot)
        print("y_true shape: ",y_true_onehot.shape)
        print("y_pred_probs: ",y_pred_probs)
        print("y_pred_probs: ",y_pred_probs.shape)
        
        N = y_true_onehot.shape[0]
        y_pred_probs_clipped = np.clip(y_pred_probs, 1e-12, 1.0)
        
        # L = - 1/N * sum(Y_true * log(Y_pred))
        loss = -np.sum(y_true_onehot * np.log(y_pred_probs_clipped)) / N
        return loss

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Calculates Mean Squared Error (MSE) Loss (for regression)."""
        
        N = y_true.shape[0]
        # L = 1/2N * sum((Y_pred - Y_true)^2)
        return np.sum((y_pred - y_true)**2) / (2 * N)

    def get_loss(self, name: str):
        """Returns the loss function reference based on the name."""
        name = name.lower()
        if name == "cross_entropy":
            return self.cross_entropy
        elif name == "mse":
            return self.mean_squared_error
        else:
            raise ValueError(f"Unknown loss function: {name}")
        
