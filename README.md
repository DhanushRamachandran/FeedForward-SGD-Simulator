# FeedForward-SGD-Simulator

This project is a high-fidelity manual simulation of a Multi-Layer Perceptron (MLP) designed for classification tasks (e.g., MNIST digit recognition). Unlike high-level frameworks like Keras or PyTorch, this simulator implements the mathematical foundations of deep learning—including forward propagation, loss computation, and backpropagation—from scratch using NumPy. It serves as a pedagogical tool to visualize weight evolution and activation changes during the training process.

## What are the components?

THe core components of any neural network are:
Simulator-INput Template
SImulator with logging and tracking of outputs
Visualizer Module

The aim in this repo is to build a modular fraamework that builds and simulates any neural network architecture that uses "Stochastic Gradient Descent" optimizer, given the proper specifications as a JSON template.
The simulator constructs and initialises the network configuration with weights and biases, activation functions, loss functions and optimizer and simulates it for a give data.

## How is the simulator framework built?
Components
The simulator is composed of the following primary modules:
Data Preprocessing Engine: Handles dataset loading (MNIST), normalization (MinMaxScaler), and label encoding (One-Hot Encoding).

Initialization Module: Manages the random initialization of weights and biases across specified layer dimensions.

Activation Functions: Implements non-linear transformations (e.g., Sigmoid, Softmax) used to introduce complexity into the model.

Optimization Engine (SGD): Executes Stochastic Gradient Descent to update network parameters based on calculated gradients.

Logging & Visualization: Captures training metrics (weight changes, activation changes) in JSON format for post-training analysis and trend visualization.

## Architecture of Neural Network
The network follows a classic Feed-Forward architecture:

Input Layer: Flat vectors representing input features (784 pixels for 28x28 images).

Hidden Layers: Fully connected layers where weights and biases are applied, followed by non-linear activation functions (64)

Output Layer: A final layer utilizing Softmax activation to produce a probability distribution across classes (10)

Cache Journey (Training Lifecycle)
During training, data undergoes a specific "journey" through the network layers:
1. Forward Propagation & Cache StorageInput Layer to Hidden Layer: Input data is multiplied by weights ($W_1$) and added to biases ($b_1$).
2. The result ($Z_1$) is passed through an activation function to produce $A_1$. This activation is "cached" for use during backpropagation.Hidden Layer to Output Layer: The process repeats ($A_1 \cdot W_2 + b_2 = Z_2$), resulting in final predictions ($\hat{y}$).
3. Loss Computation: The difference between the predicted output ($\hat{y}$) and the true label ($y$) is calculated using a loss function (typically Cross-Entropy).
4. Backpropagation & Weight UpdatesGradient Flow: The simulator calculates how much the loss changes with respect to each weight and bias, starting from the output layer and moving backwards.Layer-wise Updates: For each layer, the stored cache (activations from the forward pass) is retrieved to compute the specific weight gradients ($dW$).
5. Parameter Step: Weights and biases are updated using the SGD rule:$W = W - (\text{learning\_rate} \cdot dW)$.

## Backpropagation Algorithm
The simulator implements the chain rule of calculus to perform backpropagation. By calculating the partial derivative of the loss function with respect to each parameter, the model identifies the "direction" of steepest descent. This implementation specifically tracks:

Weight Evolution: Monitoring the average change in weights per epoch to ensure the model is converging.

Activation Stability: Tracking hidden layer activation changes to diagnose issues like vanishing or exploding gradients.

## RESULTS
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/c3ef3935-9e8a-4299-81cb-04e630eb883a" />

<img width="868" height="439" alt="image" src="https://github.com/user-attachments/assets/5fcb5edb-5420-488e-98c5-3ab29ee11475" />

<img width="1286" height="542" alt="image" src="https://github.com/user-attachments/assets/1755778e-624d-463c-a662-04e7e4dbf9c4" />



# INPUT TEMPLATE EXAMPLE:
```json
{
    "constructor_configs": {
        "layer_configs": [
            {
                "neuron_count": 784,
                "activation_function": "none"
            },
            {
                "neuron_count": 64,
                "activation_function": "relu"
            },
            {
                "neuron_count": 10,
                "activation_function": "softmax"
            }
        ],
        "loss_function": "cross_entropy",
        "optimizer": "SGD"
    },
    "training_configs": {
        "batch_size": 128,
        "learning_rate": 0.04,
        "epochs": 12
    }
}
```
# SAMPLE OUTPUT TEMPLATE (RECORDED HISTORY)
<img width="1702" height="835" alt="image" src="https://github.com/user-attachments/assets/6e830fdc-4453-41e5-a203-62cd08ebec9b" />
