'''
Created on Mar 6, 2018

@author: Justin Veyna
'''

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import os
from pickle import load
from bag_of_words_data_generator import WORD_BAG_DATA_PATH, DOC_DATA_DIR
from sense_guesser import BASE_DIR
export_dir = BASE_DIR + "bag_of_words_model.mdl"


def run_pred(d, X):
    W1 = d["W1"]
    W2 = d["W2"]
    b1 = d["b1"]
    b2 = d["b2"]
    pred = tf.tensordot(X[:145004], W1, 1)
    pred = tf.add(pred, b1)
    pred = tf.multiply(pred, X[145004:])
    pred = tf.add(pred, b2)
    pred = tf.tensordot(pred, W2, 1)
    pred = tf.sigmoid(pred)
    return pred


all_file_paths = []
for f in os.listdir(DOC_DATA_DIR):#document
    all_file_paths.append(WORD_BAG_DATA_PATH+f+".pkl")

test_X = []
test_Y = []
for i in range(33, len(all_file_paths)):
    fp = all_file_paths[i]
    with open(fp, "rb") as f:
        print(fp)
        test_x_temp, test_y_temp = load(f)
        test_X.extend(test_x_temp)
        test_Y.extend(test_y_temp)
    if len(test_X) > 1000:
        break
        

train_X = numpy.asarray(test_X)
train_Y = numpy.asarray(test_Y)
n_samples = train_X.shape[0]
print(n_samples)
with open(export_dir, "rb") as f:
    d = load(f)
    W1 = d["W1"]
    W2 = d["W2"]
    b1 = d["b1"]
    b2 = d["b2"]

X =  tf.placeholder("float")
Y = tf.placeholder("float")

pred = tf.tensordot(X[:145004], W1, 1)
pred = tf.add(pred, b1)
pred = tf.multiply(pred, X[145004:])
pred = tf.add(pred, b2)
pred = tf.tensordot(pred, W2, 1)
pred = tf.sigmoid(pred)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))



predictions = []
with tf.Session() as sess:
    for (x,y) in zip(test_X, test_Y):
        res = sess.run(pred, feed_dict={X: x, Y: y})
        predictions.append(res)
    loss = tf.losses.mean_squared_error(test_Y, predictions)
    print(sess.run(loss))

