# Simulation is executed in a modular way with records and history being collected for every layer
# , every epoch and every weight change is monitored

# DATA IS GLOBAL (x_train,x_test,y_train,y_test)
import numpy as np
# importing all the necessary libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

import json

from network_initializer import NeuralNetworkInitialser
from activation_derivatives import ActivationDerivatives
from loss_functions import LossFunction
from activation_functions import ActivationFunction
from data_preprocessing  import x_train,x_test,y_train,y_test


class NNSgdSimulator(NeuralNetworkInitialser,ActivationDerivatives,LossFunction):
    def __init__(self):
        # Neural Network constructor
        
        with open("simulator_input_template.json","r") as f:
            self.trainable_network = json.load(f)
            self.layer_configs = self.trainable_network["constructor_configs"]["layer_configs"]
            self.loss_function = self.trainable_network["constructor_configs"]["loss_function"]
            self.optimizer = self.trainable_network["constructor_configs"]["optimizer"]
            self.training_configs = self.trainable_network["training_configs"]

        super().__init__(constructor_configs=self.trainable_network["constructor_configs"],training_configs=self.training_configs)
        # Post initialization load the weights and biases into the schema 
        with open("trainable_network_template.json","r") as f:
            self.layer_configs = json.load(f)["layer_configs"]
        self.cache = dict()
        
    def forward_pass(self,X,layer_configs,epoch_count,curr_batch_no):
        """
        Performs forward propagation step through all the layers
        """
        X= X.T
        #self.cache = dict()
        if not epoch_count in self.cache.keys():
            self.cache[epoch_count] = dict()
        if not curr_batch_no in self.cache[epoch_count]:
            self.cache[epoch_count][f"batch_{curr_batch_no}"] = dict()
        self.cache[epoch_count][f"batch_{curr_batch_no}"]["A_0"] = X.tolist()

        # extraction fof weight matrix
        for i in range(0,self.no_of_layers-1):

            print("Current layer index: ",i)

            layer_config = layer_configs[i]
            
            activation_function = layer_configs[i+1]["activation_function"]
            print("activation function loaded: ",activation_function)

            w = np.array(layer_config["weights"])
            print(f"shape of W{i}_{i+1}",w.shape)
            b = np.array(layer_config["biases"])
            print(f"shape of bias",b.shape)
            # X = np.array(X)

            # applying activation function
            # output compliation

            #X = np.reshape(X,(-1,1))
            b = np.reshape(b,(-1,1))
            
            print(f"shape of bias",b.shape)
            print("shape of input: ",X.shape)
            
            Z = w@(X) + b

            # update the state history/cache
            self.cache[epoch_count][f"batch_{curr_batch_no}"][f"w_{i+1}"] = w.tolist()
            self.cache[epoch_count][f"batch_{curr_batch_no}"][f"z_{i+1}"] = Z.tolist()
            
            print("before application of activation function: ",Z.shape)

            print(f"current output after layer {i+1} computation: ",Z)

            # activation function
            act_func = ActivationFunction().get_activation(name=activation_function)
            print("activation function applied: ",act_func)

            print("Z shape: ",Z.shape)
            #Z = np.reshape(Z,(-1,1))
            print("Z shape new: ",Z.shape)
            Z_final = act_func(Z=Z)
            self.cache[epoch_count][f"batch_{curr_batch_no}"][f"A_{i+1}"] = Z_final.tolist()

            print("post application of activation function, The output is :  ",Z_final)

            Z_final_T = Z_final.T
            
            # if its the output layer update it as y_pred in state history
            #if i == 
            #self.cache[""]
            
            # updating the cache/ maintaining state history
            
            X = Z_final
            #print("the updted history is : ",self.cache)
        print("****"*8)
        return X 
        
    
    def compute_loss(self, Y_true , Y_pred,epoch_count,curr_batch_no):

        # Computes the loss and updates the state history/cache
        try:
            print("****"*8) 
            # 1. Retrieve the configured loss function name
            loss_fn = LossFunction().get_loss(name=self.loss_function) 
            print("loss function called: ",loss_fn) 
            loss = loss_fn(Y_true,Y_pred) 

            self.cache[epoch_count][f"batch_{curr_batch_no}"]["loss"] = loss

            print("loss: ",loss)

            print("Self.cache: ",self.cache)
            print("****"*8)
            
        except Exception as e:
            print("An exception occured: ",e)
    
    
    def backpropagate(self, Y_true: np.ndarray, A_L: np.ndarray,epoch_count:int,curr_batch_no) -> dict:
        
        # Performs the backpropagation step to calculate the gradients (dW and db) 
        # for all weight and bias matrices.
        #Args:
         #   Y_true (np.ndarray): True labels (one-hot encoded, shape: N_samples, N_output).
          #  A_L (np.ndarray): The network's final output activations (predictions).
           #                 Shape: (N_samples, N_output).
           # cache (dict): Intermediate values (Z, A) from the forward_pass.
        #Returns:
        #    dict: Gradients {'dW1': dW1, 'db1': db1, ...}

        print("Backpropagation starting")
        bp_history = dict()

        grads = {}
        N_samples = Y_true.shape[0]
        L = self.no_of_layers
        bp_history["grads"] = dict()
        bp_history["N_samples"]=N_samples
        bp_history["L"]=L
        
        Y_T = Y_true.T 
        bp_history["y_true"]=Y_T.tolist()
        print("Y_T: ",Y_T.shape)
        A_L_T = A_L.T # final output
        print("ALT: ",A_L_T.shape)
        bp_history["A_L_T"]=A_L_T.tolist()
        #loss_function = self.training_configs["loss_function"]
        
        
        if self.loss_function.lower() == 'cross_entropy' and self.layer_configs[L-1]["activation_function"].lower() == 'softmax':
            dZ = A_L_T - Y_T
            bp_history["dZ"]=dZ.tolist()

            
        elif self.training_configs["loss_function"].lower() == 'mean_squared_error':
            raise NotImplementedError("MSE backprop requires derivative of output activation (e.g., sigmoid_derivative).")
        else:
            raise NotImplementedError("Backprop initialization for this activation/loss combo is not implemented.")

        # 3. Loop Backward from Layer L down to Layer 1
        
        for l in reversed(range(1, L )):
            bp_history[f"layer_{l}"]=dict()
            print("current layer: ",l)
            
            # --- Gradients for Layer 2 (Parameters: W_2, b_2) ---
            
            A_prev = np.array(self.cache[epoch_count][f"batch_{curr_batch_no}"][f'A_{l-1}']).T # A_{l-1} (Input to this layer, shape: (Input_size, Samples))
            print("A_prev: ",A_prev.shape)
            bp_history["A_prev"]=A_prev.tolist()
            # 3a. Calculate dW_l (Gradient w.r.t Weights)
            # dW_l = (1/N) * dZ_l @ A_{l-1}.T 
            # dW_l shape: (Output_size, Input_size)
            dW = (1 / N_samples) * (dZ @ A_prev)
            print(dW)
            dW_true = [i for i in np.reshape(dW,(-1,1))  if i>0]
            print("dwTRue: ",dW_true)

            bp_history["dW"]=dW.tolist()

            # 3b. Calculate db_l (Gradient w.r.t Biases)
            # db_l = (1/N) * sum(dZ_l, axis=1) (sum along samples)
            # db_l shape: (Output_size,)
            db = (1 / N_samples) * np.sum(dZ, axis=1)
            bp_history["db"]=db.tolist()
            print("db: ",db.shape)

            grads[f'dW{l}'] = dW.tolist()
            grads[f'db{l}'] = db.tolist()
            
            # --- Propagate Gradient to Previous Layer (A_{l-1}) ---
            
            # We only need to compute the backpropagation step dZ_{l-1} if we are not at the input layer
            if l > 1:
                W_current = np.array(self.cache[epoch_count][f"batch_{curr_batch_no}"][f'w_{l}'])
                Z_prev = np.array(self.cache[epoch_count][f"batch_{curr_batch_no}"][f'z_{l-1}']) # Z_{l-1} (Pre-activation of the previous layer)
                activation_name_prev = self.layer_configs[l-1]["activation_function"].lower()
                
                # 3c. Calculate dA_{l-1}
                # dA_{l-1} = W_l.T @ dZ_l 
                # dA_prev shape: (Input_size, Samples)
                dA_prev = W_current.T @ dZ
                
                # 3d. Calculate dZ_{l-1}
                # dZ_{l-1} = dA_{l-1} * g'(Z_{l-1}) (Hadamard product with activation derivative)
                if activation_name_prev == 'relu':
                    dZ = dA_prev * self.relu_derivative(Z_prev) 
                elif activation_name_prev == 'sigmoid':
                    dZ = dA_prev * self.sigmoid_derivative(Z_prev) 
                # Add checks for other derivatives (tanh, etc.)
                else:
                    # This should rarely happen for a hidden layer
                    dZ = dA_prev 
                
        bp_history["grads"]=grads
        with open(r"back_prop_output.json","w") as f:
            json.dump(bp_history,f,indent=3)
        return grads
        
    def train_and_record_history(self):

        # define the params 
        batch_size = self.training_configs["batch_size"]
        learning_rate = self.training_configs["learning_rate"]
        epochs = self.training_configs["epochs"]
        print("batch size: ",batch_size)
        print("learning_rate",learning_rate)
        print("epochs",epochs)

        # training layer wise
        layer_configs = self.trainable_network["layer_configs"]
        print("loaded layer configs: ",layer_configs)

        no_of_batches = int(len(x_train)/batch_size)

        curr_epoch = 1

        while curr_epoch <= epochs:
                
            for i in range(no_of_batches-1):
                print("****"*8)
                print("****"*8)
                batch_data = x_train[i*batch_size:(i+1)*batch_size]
                print("batch_data_shape: ",batch_data.shape)
                print("Batch data going in: ")
                #for data in batch_data: # for every individaul image in MNIST
                # forward pass
                print("data_shape : ",batch_data.shape)
                y_pred = self.forward_pass(X=batch_data,layer_configs=self.layer_configs,epoch_count=curr_epoch,curr_batch_no=i+1) # state gets updted automatically
                # updated history.cache after forward pass
                print(self.cache)                    
                
                # loss computation
                y_true = y_train[i*batch_size:(i+1)*batch_size].T
                
                self.compute_loss(Y_true=y_true,Y_pred=y_pred,epoch_count=curr_epoch,curr_batch_no=i+1)
                #break
                # backpropagation
                grads=self.backpropagate(Y_true=y_true.T,A_L=y_pred.T,epoch_count=curr_epoch,curr_batch_no=i+1)

                # update the weights
                for i in range(0,int(len(grads.keys())/2)):
                    self.layer_configs[i]["weights"] = list(np.array(self.layer_configs[i]["weights"]) - self.training_configs["learning_rate"]*(np.array(grads[f"dW{i+1}"])))
                    self.layer_configs[i]["biases"] = list(np.array(self.layer_configs[i]["biases"]) - self.training_configs["learning_rate"]*(np.array(grads[f"db{i+1}"])))
                    print("weights and biases updated")

                with open("forward_pass_output.json","w") as f:
                    json.dump(self.cache,f,indent=3)
                    
                break
            curr_epoch += 1
        
    #def visualize_and_explain():

    def simulate(self):
        # begin training
        self.train_and_record_history()

# unit testing
simulator = NNSgdSimulator()
simulator.simulate()