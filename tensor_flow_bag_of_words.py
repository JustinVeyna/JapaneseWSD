'''
Created on Mar 5, 2018

@author: Justin Veyna

referenced: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from pickle import load, dump
import os
from bag_of_words_data_generator import WORD_BAG_DATA_PATH, DOC_DATA_DIR
from sense_guesser import BASE_DIR
export_dir = BASE_DIR + "bag_of_words_model.mdl"
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 3
display_step = 50

# Training Data
train_X = [] 
#numpy.asarray()
train_Y = []
all_file_paths = []
for f in os.listdir(DOC_DATA_DIR):#document
    all_file_paths.append(WORD_BAG_DATA_PATH+f+".pkl")

for fp in all_file_paths:
    with open(fp, "rb") as f:
        print(fp)
        train_x_temp, train_y_temp = load(f)
        train_X.extend(train_x_temp)
        train_Y.extend(train_y_temp)
    if len(train_X) > 1000:
        break
        

train_X = numpy.asarray(train_X)
train_Y = numpy.asarray(train_Y)
n_samples = train_X.shape[0]
#print(train_X.shape)# -> (93, 145068)

# tf Graph Input
X =  tf.placeholder("float") #145068
Y = tf.placeholder("float")

# Set model weights
W1 = tf.Variable(rng.randn(145004,64), name="weight1", dtype=numpy.float32)
W2 = tf.Variable(rng.randn(64), name="weight2", dtype=numpy.float32)
b1 = tf.Variable(rng.randn(64), name="bias1", dtype=numpy.float32)
b2 = tf.Variable(rng.randn(64), name="bias2", dtype=numpy.float32)

# Construct a linear model
#print(X.shape())
#print(X[:145005].shape())
pred = tf.tensordot(X[:145004], W1, 1)
pred = tf.add(pred, b1)
pred = tf.multiply(pred, X[145004:])
pred = tf.add(pred, b2)
pred = tf.tensordot(pred, W2, 1)
pred = tf.sigmoid(pred)



# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()



# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    '''
    for f in os.listdir(DOC_DATA_DIR):#document
       
        with open(WORD_BAG_DATA_PATH+f+".pkl", "rb") as fi:
            print(WORD_BAG_DATA_PATH+f+".pkl", "rb")
            train_X, train_Y = load(fi)
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
            # Display logs per epoch step
        if epoch >= training_epochs:
            break
        else:
            epoch += 1
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W1=", sess.run(W1), "W2=", sess.run(W2), "b1=", sess.run(b1), "b2=", sess.run(b2))
    '''
    costs = []
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
           
            # Display logs per epoch step
        if epoch >= training_epochs:
            break
        else:
            epoch += 1
            print(epoch)
        if (epoch+1) % display_step == 0:
            pass
            #c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            #    "W1=", sess.run(W1), "W2=", sess.run(W2), "b1=", sess.run(b1), "b2=", sess.run(b2))
    
    with open(export_dir, "wb") as f:
        dump({"W1":sess.run(W1), "W2":sess.run(W2), "b1": sess.run(b1), "b2":sess.run(b2)}, f)
    print("Optimization Finished!")
    
    for (x, y) in zip(train_X, train_Y):
        costs.append(sess.run(cost, feed_dict={X: x, Y: y}))
    training_cost = tf.reduce_mean(costs)
    print("Training cost=", sess.run(training_cost), '\n')
    
    '''
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    '''
    
