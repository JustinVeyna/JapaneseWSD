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
from scipy.spatial.distance import cosine, euclidean
from test_class import Test
import matplotlib.pylab as plt

BASE_DIR = 'S:/workspace/WSD/'
DOC_DATA_DIR = BASE_DIR + "jsemcor-2012-01-pickled/"
SYNSET_DATA_PATH = BASE_DIR + "data/polyglot-ja-synset-avg.pkl"
WORD2VEC_DATA_DIR = BASE_DIR + 'data/polyglot-ja.pkl'

DIFFERENCE_FORMULA = euclidean #or Cosine
CONTEXT_SIZE = "sentence"#"sentence"


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
    #returns {paragraph# : {sentence# : {word# : {"wid" : w_id(not the wordnet wordid),"text": text,"sense": sense_id(the wordnet one)}}}}
    with open(DOC_DATA_DIR + f, "rb") as s:
        words = load(s)
    return words

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
    return ret

def get_paragraph_average(word2vec_dic, p):
    paragraph_avg = numpy.zeros(64)
    count = 0
    for s in range(len(p)):#sentence
        for word in p[s].values():
            text = word["text"]
            if text in word2vec_dic:
                vec_val = get_word_vec_from_word(word2vec_dic, text)
                paragraph_avg += vec_val
                count+=1
    if count > 0:
        paragraph_avg /= count
    return count, paragraph_avg

def get_sentence_avg_vec(word2vec_dic, s):
    sentence_avg = numpy.zeros(64)
    count = 0
    for word in s.values():
        text = word["text"]
        if text in word2vec_dic:
            vec_val = get_word_vec_from_word(word2vec_dic, text)
            sentence_avg += vec_val
            count+=1
    if count > 0:
        sentence_avg /= count
    return count, sentence_avg

GET_SENSE_CALCULATED = {None: ([],[])}
def get_senses(word_to_check, synset_data):
    if word_to_check in GET_SENSE_CALCULATED:
        return GET_SENSE_CALCULATED[word_to_check]
    words = WordLoader().load_words_with_lemma(word_to_check)
    senses = []
    senses_vecs = []
    for word in words:
        #print(word)
        senses_this_ittr = SenseLoader().load_senses_with_synset(word)
        #senses_this_ittr = SenseLoader().load_senses_with_synset_plus(word)
        senses += senses_this_ittr
        senses_vecs += get_vecs_from_senses(synset_data, senses_this_ittr)
        #print(senses)
    if len(senses) > 200:
        print("This word: ", word_to_check)
        GET_SENSE_CALCULATED[word_to_check] = ([],[])
        return ([],[])
    GET_SENSE_CALCULATED[word_to_check] = (senses, senses_vecs)
    return (senses, senses_vecs)

def get_closest_sense(sentence_avg, senses_vecs):
    diffs = list(map(lambda x: DIFFERENCE_FORMULA(sentence_avg, x), senses_vecs))
    return numpy.argmin(diffs)

def run_test(test_classes, sentence_tokens, word, word2vec_dic=load_word2vec_dic(), synset_data=load_synset_data()):
    sentence_avg = get_sentence_avg_vec(word2vec_dic, sentence_tokens)
    sentence_length = len(sentence_tokens)
    for w in range(len(sentence_tokens)):#word
        word = sentence_tokens[w]
        senses, senses_vecs = get_senses(word, synset_data)  
        for test in tests:
            test.run_test(labeled_word_sense, senses, senses_vecs, sentence_avg, sentence_length, word["text"])


if __name__ == '__main__':
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    tests = [Test(euclidean, name="Euclidean")]
    max_docs = 100000
    z=0
    sanity_check = 0
    for f in os.listdir(DOC_DATA_DIR):#document
        z+=1
        if z > max_docs:
            break
        print(f)
        doc_words_dic = load_words()
        
        for para in range(len(doc_words_dic)):#paragraph
            p = doc_words_dic[para]
            if CONTEXT_SIZE == "paragraph":
                words_count, words_avg = get_paragraph_average(word2vec_dic, doc_words_dic[para])
            for sent in range(len(doc_words_dic[para])):#sentence
                if CONTEXT_SIZE == "sentence":
                    words_count, words_avg = get_sentence_avg_vec(word2vec_dic, doc_words_dic[para][sent])
                for w in range(len(doc_words_dic[para][sent])):#word
                    word = doc_words_dic[para][sent][w]
                    labeled_word_sense = word["sense"]
                    senses, senses_vecs = get_senses(word["text"], synset_data)  
                    for test in tests:
                        test.run_itteration(labeled_word_sense, senses, senses_vecs, words_avg, words_count, word["text"])
                        
    
    for test in tests:
        print(sanity_check)
        test.print_results()
        test.print_sense_details()
        #test.print_sense_ignorance_dict()
        word_accuracy = test.get_word_accuracy()
        num_to_print = 5
        print(word_accuracy[0:num_to_print],word_accuracy[-num_to_print:])
        to_plt = test.get_plotable_sentence_length_histogram()
        y,x = list(zip(*to_plt))
        plt.plot(y,x)
        plt.suptitle(test.name)
        plt.show()
                        
                    
                    
                    
                    
                    