'''
Created on Feb 21, 2018

@author: Justin Veyna
'''

from synset_avg_generator import synset_entry
from xml_parser import ddict, dddict, ddddict
from sense_guesser import *#load_word2vec_dic, load_synset_data, DOC_DATA_DIR, load_words, get_senses, get_word_vec_from_word
import os
import numpy
MAX_DOCS = 1000000000

def make_sentence_array(word2vec_dic, sentence):
    ret = []
    for w in range(len(sentence)):
        word = sentence[w]
        if word["text"] in word2vec_dic:
            word_array = get_word_vec_from_word(word2vec_dic, word["text"])
            #print(ret, word_array)
            ret.append(word_array)
    return ret

def zero_out_word(sent_arr, word_vec):
    word_omitted_sentence = sent_arr[:]
    word_index = word_omitted_sentence.index(word_vec)
    word_omitted_sentence[word_index]=numpy.zeros(numpy.size(word_omitted_sentence[word_index]))
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
                sent_arr = make_sentence_array(word2vec_dic, s)
                for w in range(len(s)):#word
                    word = s[w]
                    labeled_word_sense = word["sense"]
                    senses, sense_vecs = get_senses(word["text"], synset_data)
                    if word["text"] in word2vec_dic:
                        word_array = get_word_vec_from_word(word2vec_dic, word["text"])
                        for i in range(len(senses)):
                            sense = senses[i]
                            sense_vec = sense_vecs[i]
                            word_omitted_sentence = zero_out_word(sent_arr, word_array)
                            sentence_with_sense = word_omitted_sentence.append(sense_vec)
                            data_x.append(sentence_with_sense)
                            if sense == labeled_word_sense:
                                #correct
                                data_y.append(1)
                            else:
                                #incorrect
                                data_y.append(0)
        print(data_x, data_y)