'''
Created on Jan 30, 2018

@author: Justin Veyna
'''
import pickle
import numpy
#import sys
from word_loader import WordLoader
from sense_loader import SenseLoader
from collections import defaultdict
from numpy import float32

ja_encs = ["cp932", "euc_jp","euc_jis_2004","euc_jisx0213","iso2022_jp","iso2022_jp_1","iso2022_jp_2","iso2022_jp_2004","iso2022_jp_3","iso2022_jp_ext","shift_jis","shift_jis_2004","shift_jisx0213","utf_32","utf_32_be","utf_32_le","utf_16","utf_16_be","utf_16_le","utf_7","utf_8","utf_8_sig"]
enc ="bytes"
BASE_DIR = 'S:/workspace/WSD/'
DATA_DIR = BASE_DIR + 'data/polyglot-ja.pkl'
SAVE_DIR = BASE_DIR + "data/polyglot-ja-synset-avg.pkl"
SPECIAL_SYMBOL_COUNT = 4


def synset_entry():
    return {"count":0, "embedding":numpy.zeros(64, float32)}
SYNSET_VEC_AVG = defaultdict(synset_entry)

if __name__ == '__main__':
    #print(sys.stdout.encoding)
    words, embeddings = pickle.load(open(DATA_DIR, 'rb'), encoding=enc)
    #print("Emebddings shape is {}".format(embeddings.shape))
    num_words = embeddings.shape[0]
    #print(words[0:100])
    #print(embeddings[0:100])
    
    for i in range(SPECIAL_SYMBOL_COUNT, num_words):
        if i%int(num_words/20) == 0:
            print("{}/{}".format(i,num_words))
        lemma = words[i]
        embedding = embeddings[i]
        lemma_entries = WordLoader().load_words_with_lemma(lemma)
        for word in lemma_entries:
            senses = SenseLoader().load_senses_with_synset(word)
            #print("{:<80}".format(str(word)), "sense count: ", len(senses))
            for sense in senses:
                SYNSET_VEC_AVG[sense.synset]["count"] += 1
                SYNSET_VEC_AVG[sense.synset]["embedding"] += embedding
    '''
    for key, value in SYNSET_VEC_AVG.items():
        print(value)
    '''
    with open(SAVE_DIR, "wb") as f:
        pickle.dump(dict(SYNSET_VEC_AVG),f)
                
        
        
        