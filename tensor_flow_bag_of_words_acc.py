'''
Created on Mar 6, 2018

@author: Justin Veyna
'''


import tensorflow as tf
from bag_of_words_data_generator import *
import numpy as np
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
    '''
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
    '''
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    z = 0
    correct = 0
    total = 0
    with tf.Session() as sess:
        for f in os.listdir(DOC_DATA_DIR):#document
            z+=1
            if z > MAX_DOCS:
                break
            print(f)
            doc_words_dic = load_words(f)
            sentence_count = 0
            data_x = []
            data_y = []
            for para in range(len(doc_words_dic)):#paragraph
                p = doc_words_dic[para]
                for _ in range(len(doc_words_dic[para])):#sentence
                    sentence_count += 1
                    print(sentence_count)
                    s = doc_words_dic[para][sentence_count]
                    sent_arr, correctness = word_bag_make_sentence_array(word2vec_dic, synset_data, s, form=True)
                    ind=0
                    for i in range(len(sent_arr)):
                        sense_closeness =[]
                        correct_index = 0
                        for j in range(len(sent_arr[i])):
                            score = (correctness[ind]*-1) * -sess.run(run_pred(d, np.asarray(sent_arr[i][j]).astype("float32")))
                            sense_closeness.append(score)
                            if correctness[ind]==1:
                                correct_index = j
                            ind+=1
                        if len(sense_closeness) > 0:
                            if sense_closeness.index(max(sense_closeness)) == correct_index:
                                correct+=1
                            total+=1
        print(correct, total)
        print(correct/total)
                            
                    