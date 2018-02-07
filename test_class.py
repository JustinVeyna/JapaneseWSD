'''
Created on Feb 4, 2018

@author: Justin Veyna
'''
import numpy
from collections import defaultdict
from random import Random
MINIMUN_SENSE_COUNT = 1
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

def get_closest_sense(sentence_avg, senses, senses_vecs, difference_function):
    diffs = list(map(lambda x: difference_function(sentence_avg, x), senses_vecs))
    diffs = sorted(zip(senses, diffs), key = lambda x: x[1])
    #print(diffs)
    return diffs


class Test():
    '''
    classdocs
    '''
    def __init__(self, difference_function, name="", sense_ignorance_check=False, random=False):
        '''
        Constructor
        '''
        self.difference_function = difference_function
        self.correct = 0
        self.total = 0
        self.senses_total = 0
        self.name = name
        self.sentence_length_histogram = defaultdict(lambda: {"count":0, "correct":0})
        self.word_histogram = defaultdict(lambda: {"count":0, "correct":0})
        self.rank_history = []
        self.sense_ignorance_count = 0
        self.sense_ignorance_dict = defaultdict(lambda: [int(), defaultdict(int)])
        self.sense_ignorance_check  = sense_ignorance_check
        self.random = random
        
    def run_itteration(self, labeled_word_sense, senses, senses_vecs, sentence_avg, sentence_length = 0, word = ""):
        if labeled_word_sense != None and len(senses_vecs) >= MINIMUN_SENSE_COUNT:
            self.sense_ignorance_logging(labeled_word_sense, word, senses)
            self.senses_total += len(senses)
            self.total += 1
            self.sentence_length_histogram[sentence_length]["count"]+=1
            self.word_histogram[word]["count"]+=1
            
            sense_guesses = get_closest_sense(sentence_avg, senses, senses_vecs, self.difference_function)
            if self.random:
                Random().shuffle(sense_guesses)
            correct_sense_rank = self._get_correct_sense_rank(labeled_word_sense, sense_guesses)
            self.rank_history.append(correct_sense_rank)
            
            if sense_guesses[0][0].synset == labeled_word_sense:#if correct
                self.correct += 1
                self.sentence_length_histogram[sentence_length]["correct"]+=1
                self.word_histogram[word]["correct"]+=1

    def _get_correct_sense_rank(self, labeled_word_sense, sense_guesses):
        for i in range(len(sense_guesses)):
            if sense_guesses[i][0].synset == labeled_word_sense:
                return i+1
        return len(sense_guesses)+1

    def sense_ignorance_logging(self, labeled_word_sense, word, senses):
        if self.sense_ignorance_check and not labeled_word_sense in map(lambda x: x.synset,senses):
            self.sense_ignorance_count+=1
            self.sense_ignorance_dict[word][0]+=1
            self.sense_ignorance_dict[word][1][labeled_word_sense]+=1
    
    def get_results(self):
        accuracy = self.correct/self.total*100.0
        ret = "{}/{} correct {}%, MRR: {}".format(self.correct, self.total, accuracy, self.get_mrr())
        if self.name != "":
            ret = "{}: \n".format(self.name) + ret
        return ret
    
    def print_results(self):
        print(self.get_results())
    
    def get_mrr(self):
        rh_len = len(self.rank_history)
        total = 0.0
        for i in range(rh_len):
            total+= 1.0/self.rank_history[i]
        return total/rh_len
    
    def get_plotable_sentence_length_histogram(self):
        tmp = sorted(list(self.sentence_length_histogram.items()))
        ret = []
        for val, d in tmp:
            ret.append((val, d["correct"]/d["count"]))
        return ret

    def get_word_accuracy(self):
        tmp = sorted(list(self.word_histogram.items()), reverse=True, key = lambda x: x[1]["correct"]/x[1]["count"])
        ret = []
        for val, d in tmp:
            ret.append((val, d["correct"]/d["count"]))
        return ret
    
    def get_sense_details(self):
        return "{} senses (avg: {}), Sense ignorance count:{}".format(self.senses_total, self.senses_total/self.total, self.sense_ignorance_count)
    
    def print_sense_details(self):
        print(self.get_sense_details())
        
    def print_sense_ignorance_dict(self):
        print(sorted(self.sense_ignorance_dict.items(), key = lambda x: x[1][0], reverse=True)[:10])