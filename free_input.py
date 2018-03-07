'''
Created on Feb 23, 2018

@author: Justin Veyna
'''
from scipy.spatial.distance import cosine, euclidean
from sense_guesser import get_senses, load_word2vec_dic, load_synset_data
from synset_avg_generator import synset_entry
import tinysegmenter
import numpy
from synset_def_loader import SynsetDefLoader

class Sentence():
    def __init__(self, word2vec_dic, synset_data, raw_sentence):
        self.word2vec_dic = word2vec_dic
        self.synset_data = synset_data
        self.raw_sentence = raw_sentence
        self.difference_function = euclidean
        self.tokenizer = tinysegmenter.TinySegmenter()
        self._tokenize_sentence()
        self._precalculations()
    
    def _tokenize_sentence(self):
        self.tokens = self.tokenizer.tokenize(self.raw_sentence)
        self.word_count=len(self.tokens)
        
    def _sentence_avg_vec(self):
        sentence_avg = numpy.zeros(64)
        count = 0
        for word in self.tokens:
            if word in self.word2vec_dic:
                vec_val = self.word2vec_dic[word]
                sentence_avg += vec_val
                count+=1
        return count, sentence_avg
    
    def _precalculations(self):
        #create avg sentence vec or for CNN create ...
        self.sentence_vec_count, self.sentence_sum_vec = self._sentence_avg_vec()
    
        
    def _sense_ranker(self, senses_vecs, senses, word_vec):
        calculated_avg = (self.sentence_sum_vec - word_vec)/self.sentence_vec_count
        scores = list(map(lambda x: self.difference_function(calculated_avg, x), senses_vecs))
        sorted_scores = sorted(list(zip(scores, senses)), reverse=True)
        return sorted_scores
    
    def guess_sense(self, index = None, word=None):
        #give sense rank
        sense_guesses = []
        if index != None:
            word = self.tokens[index]
        print(word)
        if word in self.word2vec_dic:
            word_vec = self.word2vec_dic[word]
            senses, senses_vecs = get_senses(word, self.synset_data)
            sense_guesses = self._sense_ranker(senses_vecs, senses, word_vec)
        #print(sense_guesses)
        return sense_guesses

def sentence_to_output(sentence):
    output = ""
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    s = Sentence(word2vec_dic, synset_data, sentence)
    for i in range(s.word_count):
        output += "\n" + s.tokens[i] + "<br/>\n"
        sense_guesses = s.guess_sense(i)
        for score, sense in sense_guesses:
            sense_definition = SynsetDefLoader().load_syndef_with_sense(sense)
            output +=str(score) + str([x.defin for x in sense_definition]) + "<br/>"
    return output
if __name__ == '__main__':
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    sentence =  "こんにちは、私の声は聞こえますか。"#input()
    s = Sentence(word2vec_dic, synset_data, sentence)
    for i in range(s.word_count):
        sense_guesses = s.guess_sense(i)
        for score, sense in sense_guesses:
            sense_definition = SynsetDefLoader().load_syndef_with_sense(sense)
            print(score, [x.defin for x in sense_definition])
    