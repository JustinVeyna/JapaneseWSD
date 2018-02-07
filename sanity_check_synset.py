'''
Created on Feb 6, 2018

@author: Justin Veyna
'''
from sense_loader import SenseLoader
from synset_loader import SynsetLoader
from synlink_loader import SynlinkLoader
from word_loader import WordLoader
from random import Random
from synset_avg_generator import synset_entry
from xml_parser import ddict, dddict, ddddict

from sense_guesser import load_word2vec_dic, load_synset_data, get_vecs_from_senses
import numpy
from test_class import get_closest_sense
from scipy.spatial.distance import cosine, euclidean


def get_similar_synset_ids(synset):
    synlinks = []
    for link in ["also","sim"]:
        synlinks += SynlinkLoader().load_synlinks_with_sense_and_link(synset, link)
    return list(map(lambda x: x.synset2, synlinks))  

def average_synset_vectors(synset_ids, synset_data):
    total = numpy.zeros(64)
    count = 0
    for ssi in synset_ids:
        if ssi in synset_data and synset_data[ssi]["count"]:
            total+=(synset_data[ssi]["embedding"]/synset_data[ssi]["count"])
            count+=1
    if count == 0:
        return total
    return total/count

if __name__ == '__main__':
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    total = 0
    correct = 0
    
    for wordid in range(155288,249121):
        if wordid%1000 ==0:
            print(wordid)
        word = WordLoader().load_word_with_wordid(wordid)
        lemma_senses = SenseLoader().load_senses_with_synset(word)
        lemma_senses_vecs = get_vecs_from_senses(synset_data, lemma_senses)
        
        chosen_sense_index = Random().randint(0, len(lemma_senses)-1)
        chosen_synset_id = lemma_senses[chosen_sense_index].synset
        synset = SynsetLoader().load_synset_with_synset(chosen_synset_id)
        similar_synset_ids = get_similar_synset_ids(synset)
        similar_synset_avg_vector = average_synset_vectors(similar_synset_ids, synset_data)
        
        sense_ranking = get_closest_sense(similar_synset_avg_vector, lemma_senses, lemma_senses_vecs, euclidean)
        
        if sense_ranking[0][0].synset == chosen_synset_id:#correct
            correct+=1
        total +=1
    
    print(correct,total, correct/total)
    
    
    
        