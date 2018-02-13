
# Based on a tutorial from here: https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow
# First we need to import the data which is saved as a CSV in the same file called data_stocks.csv

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import data
data = pd.read_csv('Goals_prediction.csv')

#Check rows and columns of total CSV data
no_rows = data.shape[0]
no_columns = data.shape[1]
print("number of rows: ", no_rows)
print("number of columns: ", no_columns)

#Reduce the data to the columns we care about
data = data[['Goals scored - actual', 'Goals scored last game', 'Average goals scored per game',
'Goals scored on average for home or away', 'Average goals conceded by opponent this season']]

#Make the datafile a numpy array
data = data.values

#Split the data into test set and training set
test_set_percentage = 0.2
data_test = data[:int(test_set_percentage*no_rows),:]
data_train = data[int((test_set_percentage)*no_rows):,:]

"""
#Scale the data so that is is better for deep learning
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

"""

#Split training data into x and y
y_train = data_train[:,0]
x_train = data_train[:,1:]

#Split test data into x and y
y_test = data_test[:,0]
x_test = data_test[:,1:]

#Scale the data so that is is better for deep learning
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Check the shapes of the different arrays
x_train_shape = x_train.shape
y_train_shape = y_train.shape
x_test_shape = x_test.shape
y_test_shape = y_test.shape

print("x_train shape is: ", x_train_shape)
print("y_train shape is: ", y_train_shape)
print("x_test shape is: ", x_test_shape)
print("y_test shape is: ", y_test_shape)


# Define inference graph in tensorflow based on 4 hidden layers in a neural network

#Define the number of input features
print("The number of features is: ", x_train.shape[1])
n_input_features = x_train.shape[1]

#Define the number of nodes in each hidden layer
n_neurons_1 = 512
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

#Define the number of nodes in the output layer
n_target = 1

#Create placeholders for x and y
x = tf.placeholder(tf.float32, shape = [None, n_input_features], name = "x")
y = tf.placeholder(tf.float32, shape = [None], name = "y")

# Setup initilizers as per the xavier initlizers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


# Layer 1: Variables for hidden weights and biases
w1 = tf.Variable(weight_initializer([n_input_features, n_neurons_1]))
b1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
w2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
b2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
w3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
b3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
w4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
b4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
w5 = tf.Variable(weight_initializer([n_neurons_4, n_target]))
b5 = tf.Variable(bias_initializer([n_target]))

#Forward propagation
#Use relu functions for forward propagation through the layers

a1 = tf.nn.relu(tf.add(tf.matmul(x,w1), b1))
a2 = tf.nn.relu(tf.add(tf.matmul(a1,w2), b2))
a3 = tf.nn.relu(tf.add(tf.matmul(a2,w3), b3))
a4 = tf.nn.relu(tf.add(tf.matmul(a3,w4), b4))

print(a4.shape)
print(w5.shape)
print(b5.shape)

#Forward propagation to the output, it needs to be transposed as the relu function aboves transposes for the others
prediction = tf.transpose(tf.add(tf.matmul(a4, w5), b5))
print(prediction)

# Define the cost function

#Use the squared difference cost function
cost = tf.reduce_mean(tf.squared_difference(prediction, y))

#Optimizer
opt = tf.train.AdamOptimizer().minimize(cost)

# Run the session

#Initialize variables
init = tf.global_variables_initializer()

# Make Session
sess = tf.Session()

# Run initializer
sess.run(init)

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# Number of epochs and batch size
epochs = 10
batch_size = 10

for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    
    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        
        # Run optimizer with batch
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        
        # Show progress
        if np.mod(i, 20) == 0:
            # Prediction
            pred = sess.run(prediction, feed_dict={x: x_test})
            #Populate chart
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)

# Print final cost after Training
final_cost = sess.run(cost, feed_dict={x: x_test, y: y_test})
print("The final cost is: ",final_cost)

final_predictions = sess.run(prediction, feed_dict={x: x_test})
print(final_predictions)


