'''
Created on Feb 4, 2018

@author: Justin Veyna
'''
import numpy
from collections import defaultdict
'''
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
                    senses, senses_vecs = get_senses(word)  
                    if labeled_word_sense != None and len(senses_vecs) > 1:
                        senses_total += len(senses)
                        total += 1
                                                
                        guessed_sense_index = get_closest_sense(sentence_avg, senses_vecs)
                        guessed_sense = senses[guessed_sense_index]
                        
                        if guessed_sense.synset == labeled_word_sense:
                            correct += 1
'''

def get_closest_sense(sentence_avg, senses_vecs, difference_function):
    diffs = list(map(lambda x: difference_function(sentence_avg, x), senses_vecs))
    return numpy.argmin(diffs)

class Test():
    '''
    classdocs
    '''
    def __init__(self, difference_function, qualification_function=None, name=""):
        '''
        Constructor
        '''
        self.difference_function = difference_function
        self.qualification_function = qualification_function
        self.correct = 0
        self.total = 0
        self.senses_total = 0
        self.sentence_length_histogram = defaultdict(lambda: {"count":0, "correct":0})
        self.name = name
        
        
    def run_itteration(self, labeled_word_sense, senses, senses_vecs, sentence_avg, sentence_length = 0):
        if labeled_word_sense != None and len(senses_vecs) > 1:
            self.senses_total += len(senses)
            self.total += 1
            self.sentence_length_histogram[sentence_length]["count"]+=1
            guessed_sense_index = get_closest_sense(sentence_avg, senses_vecs, self.difference_function)
            guessed_sense = senses[guessed_sense_index]
            if guessed_sense.synset == labeled_word_sense:
                self.correct += 1
                self.sentence_length_histogram[sentence_length]["correct"]+=1
    def get_results_string(self):
        accuracy = self.correct/self.total*100.0
        ret = "{}/{} correct {}%".format(self.correct, self.total, accuracy)
        if self.name != "":
            ret = "{}: ".format(self.name) + ret
        return ret
    
    def print_results(self):
        print(self.get_results_string())
        