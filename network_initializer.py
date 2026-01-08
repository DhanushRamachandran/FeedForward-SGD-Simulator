# network constructor
# importing all the necessary libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

import json

class NeuralNetworkInitialser():
    """
    Parent class responsible for defining the network architecture 
    and initializing all weight and bias parameters.
    retuns a trainable network configuration with weights and bias values
    """
    def __init__(self, constructor_configs:dict,training_configs:dict):
        
        self.layer_configs = constructor_configs["layer_configs"]
        self.no_of_layers = len(self.layer_configs)
        self.training_configs = training_configs

        # Dictionary to store all parameters (W1, b1, W2, b2, etc.)
        self.trainable_network =  constructor_configs
        #self.w = {"0":{}}
        #self.b = {"0":{}}
                
        print(f"--- Architecture Constructor Initialized ---")
        print(f"Schema: {self.layer_configs} (Input -> Hidden(s) -> Output)")
        
        self.initialise_and_save()
        
    def _initialize_parameters(self, weight_scale=0.03):
        """
        Initializes weights (W) and biases (b) for all connections.
        Weights are initialized using small random values (Heuristic).
        """
        # Loop runs for the number of connections/computational layers (N-1)
        for layer_no in range(self.no_of_layers - 1):
            
            input_dim = self.layer_configs[layer_no]["neuron_count"]
            output_dim = self.layer_configs[layer_no+1]["neuron_count"]
            #layer_index = i + 1  # 1-based index for parameters (W1, b1, etc.)

            # W shape: (output_dim, input_dim)
            W = np.random.randn(output_dim, input_dim) * weight_scale

            print("weights--sample: ",W[5])
            
            # b shape: (output_dim,)
            b = np.zeros(output_dim)

            #self.w["0"][str(layer_no)] = W
            #self.b["0"][str(layer_no)] = b
            
            print("Bias sample: ",b[:5])

            self.trainable_network["layer_configs"][layer_no][f"weights"] = W.tolist()
            self.trainable_network["layer_configs"][layer_no]["biases"] = b.tolist()
            
            print(f"Layer {layer_no+1} weights and biases: W{layer_no+1} ({W.shape}), b{layer_no+1} ({b.shape})")
            
        print("--- Initialization Complete ---")

    def save_trainable_network(self):
        with open("trainable_network_template.json","w") as f:
            json.dump(self.trainable_network,f,indent=3)

    def initialise_and_save(self):

        self._initialize_parameters()
        self.trainable_network["training_configs"] = self.training_configs
        self.save_trainable_network()
        
        
if __name__=="__main__":

    with open("simulator_input_template.json","r") as f:
        simulator_template = json.load(f)
        print("simulator template loaded successfully")
        network_initialiser = NeuralNetworkInitialser(constructor_configs=simulator_template["constructor_configs"],training_configs=simulator_template["training_configs"])
        