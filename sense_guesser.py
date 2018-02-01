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
from scipy.spatial.distance import cosine, euclidean


BASE_DIR = 'S:/workspace/WSD/'
DOC_DATA_DIR = BASE_DIR + "jsemcor-2012-01-pickled/"
SYNSET_DATA_PATH = BASE_DIR + "data/polyglot-ja-synset-avg.pkl"
WORD2VEC_DATA_DIR = BASE_DIR + 'data/polyglot-ja.pkl'

DIFFERENCE_FORMULA = euclidean


def get_word_vec_from_word(word2vec_dic, word):
    return word2vec_dic[word]

def get_vec_from_sense(sense_dic, sense):
    sense_id = sense.synset
    sense_entry = sense_dic[sense_id]
    if sense_entry["count"] > 0:
        sense_vec = sense_entry["embedding"]/sense_entry["count"]
    else:
        return numpy.zeros(64)
    return sense_vec

def get_vecs_from_senses(sense_dic, senses):
    ret = []
    for sense in senses:
        ret.append(get_vec_from_sense(sense_dic, sense))
    '''
    if len(senses) != len(ret):
        print(len(senses), len(ret))
        print(senses)
        print(ret)
    '''    
    return ret

def get_sentence_avg_vec(word2vec_dic, s):
    sentence_avg = numpy.zeros(64)
    count = 0
    for word in s.values():
        text = word["text"]
        try:
            vec_val = get_word_vec_from_word(word2vec_dic, text)
            sentence_avg += vec_val
            count+=1
        except KeyError:
            pass
    if count > 0:
        sentence_avg /= count
    return sentence_avg

def load_synset_data():
    #returns {synset_id: {"count" : int, "embedding" : embedding_vec_summed}}
    with open(SYNSET_DATA_PATH, "rb") as s:
        synset_data = load(s)
    return synset_data

def load_word2vec_dic():
    #returns {wid: embedding_vec}
    words, embeddings = load(open(WORD2VEC_DATA_DIR, 'rb'), encoding="bytes")
    word2vec_dic = dict(zip(words, embeddings))
    return word2vec_dic

def load_words():
    #returns {paragragh# : {sentence# : {word# : {"wid" : w_id(not the wordnet wordid),"text": text,"sense": sense_id(the wordnet one)}}}}
    with open(DOC_DATA_DIR + f, "rb") as s:
        words = load(s)
    return words

if __name__ == '__main__':
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    correct = 0
    total = 0
    senses_total = 0
    for f in os.listdir(DOC_DATA_DIR):#document
        print(f)
        doc_words_dic = load_words()
        
        for p in range(len(doc_words_dic)):#paragraph
            p = doc_words_dic[p]
            
            for s in range(len(p)):#sentence
                s = p[s]
                
                sentence_avg = get_sentence_avg_vec(word2vec_dic, s)

                for w in range(len(s)):#word
                    word = s[w]
                    labeled_word_sense = word["sense"]
                    wordtuple = namedtuple("word", "wid")
                    wordtuple.wid = word["wid"]
                    #print(word["wid"])
                    
                    words = WordLoader().load_words_with_lemma(word["text"])
                    senses = []
                    senses_vecs = []
                    for word in words:
                        #print(word)
                        senses_this_ittr = SenseLoader().load_senses_with_synset(word)
                        senses += senses_this_ittr
                        senses_vecs += get_vecs_from_senses(synset_data, senses_this_ittr)
                        #print(senses)
                        
                    if labeled_word_sense != None and len(senses_vecs) > 1:
                        cosine_diffs = list(map(lambda x: DIFFERENCE_FORMULA(sentence_avg, x), senses_vecs))
                        guessed_sense_index = numpy.argmin(cosine_diffs)
                        #print(guessed_sense_index)
                    else:
                        continue
                    senses_total += len(senses)
                    guessed_sense = senses[guessed_sense_index]
                    total += 1
                    if guessed_sense.synset == labeled_word_sense:
                        correct += 1
                        #print(correct, ": of {} senses, and {} sense vectors".format(len(senses), len(senses_vecs)))
                    else:
                        pass#print(guessed_sense.synset, labeled_word_sense)
    print("{}/{} correct {}%".format(correct, total, correct/total*100.0))
    #print("{} senses (avg: {})".format(senses_total, senses_total/total))
                        
                    
                    
                    
                    