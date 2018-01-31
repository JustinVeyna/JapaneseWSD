'''
Created on Jan 30, 2018

@author: Justin Veyna
'''
from pickle import load
import os

BASE_DIR = 'S:/workspace/WSD/'
DOC_DATA_DIR = BASE_DIR + "jsemcor-2012-01-pickled/"
SYNSET_DATA_PATH = BASE_DIR + "data/polyglot-ja-synset-avg.pkl"


if __name__ == '__main__':
    synset_data = None
    with open(SYNSET_DATA_PATH, "rb") as s:
        synset_data = load(s)
    for f in os.listdir(DOC_DATA_DIR):
        print(f)
        words = None
        with open(DOC_DATA_DIR + f, "rb") as s:
            words = load(s)