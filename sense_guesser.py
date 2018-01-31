'''
Created on Jan 30, 2018

@author: Justin Veyna
'''
from pickle import load
import os
import numpy
from word_loader import WordLoader
from sense_loader import SenseLoader
from synset_avg_generator import synset_entry
from xml_parser import ddict, dddict, ddddict
from collections import namedtuple

BASE_DIR = 'S:/workspace/WSD/'
DOC_DATA_DIR = BASE_DIR + "jsemcor-2012-01-pickled/"
SYNSET_DATA_PATH = BASE_DIR + "data/polyglot-ja-synset-avg.pkl"
WORD2VEC_DATA_DIR = BASE_DIR + 'data/polyglot-ja.pkl'


def get_word_vec_from_word(word2vec_dic, word):
    return word2vec_dic[word]

if __name__ == '__main__':
    words, embeddings = load(open(WORD2VEC_DATA_DIR, 'rb'), encoding="bytes")
    word2vec_dic = dict(zip(words, embeddings))
    
    synset_data = None
    with open(SYNSET_DATA_PATH, "rb") as s:
        synset_data = load(s)
    for f in os.listdir(DOC_DATA_DIR):
        print(f)
        words = None
        with open(DOC_DATA_DIR + f, "rb") as s:
            words = load(s)
        for p in range(len(words)):#paragraph
            for s in range(len(words[p])):#sentence
                sentence_avg = numpy.zeros(64)
                count = 0
                for w in range(len(words[p][s])):#word
                    text = words[p][s][w]["text"]
                    try:
                        vec_val = get_word_vec_from_word(word2vec_dic, text)
                        sentence_avg += vec_val
                        count+=1
                    except KeyError:
                        pass
                if count > 0:
                    sentence_avg /= count
                    print(sentence_avg)
                
                for w in range(len(words[p][s])):#word
                    word = words[p][s][w]
                    wordtuple = namedtuple("word", "wid")
                    wordtuple.wid = word["wid"]
                    word = WordLoader().load_word_with_wordid(wordtuple)
                    senses = map(lambda x: x.synset, SenseLoader().load_senses_with_synset(word))
                    print("here: ", senses)
                    