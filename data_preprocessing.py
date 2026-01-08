# importing all the necessary libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import json

# load dataset
dataset = mnist.load_data()
print(dataset)
print("dataset type: ",type(dataset))
y = np.array(dataset[1][1])
dataset = np.array(dataset[1][0])
#print(dataset)
print("shape of data: ",dataset.shape)
print("It has 10 K records of digits with each digit being represented in a 28X28 array")
print("Y data: ",y.shape)

unique_values,count = np.unique(y,return_counts=True)
print("distribution of y data: ",(unique_values,count))

print("From the distribution we can conclude that the dataset is pretty balanced and that we can proceed with the development of custom neural network for classification")


# data preprocessing
x=dataset
x = np.array([np.reshape(x_i,(-1,1)) for x_i in x])
x = np.reshape(x,(-1,784))

print(x.shape)
# data scaling - MinMax scaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X=x)
# data split into train,test
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state=42,test_size=0.3)

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
print(x_train[1])

# y one hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1,1))

pos = [i for i in np.reshape(x[1],(-1,1)) if i<0]
print(pos)
print(len(pos))

# sample dataset visualization
plt.figure()
sns.heatmap(data=np.reshape(x_train[10],(28,28)))
plt.show()