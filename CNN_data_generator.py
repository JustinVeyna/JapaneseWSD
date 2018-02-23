'''
Created on Feb 21, 2018

@author: Justin Veyna
'''

from synset_avg_generator import synset_entry
from xml_parser import ddict, dddict, ddddict
from sense_guesser import *#load_word2vec_dic, load_synset_data, DOC_DATA_DIR, load_words, get_senses, get_word_vec_from_word
import os
import numpy
from pickle import dump
MAX_DOCS = 1000000000

CNN_DATA_PATH = BASE_DIR + "data/cnn_data.pkl"

def merge_data(sentence_array,words_senses):
    ret = []
    for i in range(len(sentence_array)):
        word_omitted_sentence = zero_out_word(sentence_array, i)
        for sense_vec in words_senses[i]:
            ret.append(word_omitted_sentence+[sense_vec])
    return ret
            

def word_to_senses(synset_data, word):
    word_senses = []
    correctness = []
    labeled_word_sense = word["sense"]
    if labeled_word_sense != None:
        labeled_word_sense = labeled_word_sense
        senses, sense_vecs = get_senses(word["text"], synset_data)
        for i in range(len(senses)):
            sense = senses[i]
            sense = sense.synset
            #print(sense, labeled_word_sense)
            word_senses.append(sense_vecs[i])#type
            if sense == labeled_word_sense:
                #correct
                correctness.append(1)
            else:
                #incorrect
                correctness.append(0)
    return numpy.array(word_senses), numpy.array(correctness) #list


def make_sentence_array(word2vec_dic, synset_data, sentence):
    sentence_array = []
    words_senses = []
    correctness = []
    for w in range(len(sentence)):
        word = sentence[w]
        if word["text"] in word2vec_dic:
            word_array = get_word_vec_from_word(word2vec_dic, word["text"])
            sentence_array.append(word_array)#list
            
            word_senses, correctness_temp = word_to_senses(synset_data, word)
            words_senses.append(word_senses)
            correctness.extend(correctness_temp)
            
    merged_data = merge_data(sentence_array,words_senses)
    '''
    if 1 in correctness:
        print(correctness)
    else:
        print("NOTHING!")
    '''
    return merged_data, correctness

def zero_out_word(sent_arr, word_index):
    word_omitted_sentence = sent_arr[:]
    #word_index = word_omitted_sentence.index(word_vec)
    word_omitted_sentence[word_index]=numpy.zeros(numpy.size(word_omitted_sentence[word_index]), dtype=float)#list
    return word_omitted_sentence

if __name__ == '__main__':
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    data_x = []
    data_y = []
    z = 0
    for f in os.listdir(DOC_DATA_DIR):#document
        z+=1
        if z > MAX_DOCS:
            break
        print(f)
        doc_words_dic = load_words(f)
        sentence_count = 0
        for para in range(len(doc_words_dic)):#paragraph
            p = doc_words_dic[para]
            for _ in range(len(doc_words_dic[para])):#sentence
                sentence_count += 1
                s = doc_words_dic[para][sentence_count]
                sent_arr, correctness = make_sentence_array(word2vec_dic, synset_data, s)
                data_x.extend(sent_arr)
                data_y.extend(correctness)
    data_x = numpy.array(data_x)
    data_y = numpy.array(data_y)
    with open(CNN_DATA_PATH, "wb") as f:
        dump((data_x,data_y), f)
    print(data_x[0], data_y[0])