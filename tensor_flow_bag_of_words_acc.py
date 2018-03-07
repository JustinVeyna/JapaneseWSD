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
from heapdict import doc
from nltk.parse.featurechart import sent
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
if __name__ == '__main__':
    with open(export_dir, "rb") as f:
        d = load(f)
    
    for doc:
        for sent:
            #make bag or words
            for word:
                #sense_vecs = [[][][][]]
                #results = []
                for sense:
                    #test 
                    #append result
                #select highest result
                #if correct then correct+=1
                #total +=1
    #Calculate acc
                