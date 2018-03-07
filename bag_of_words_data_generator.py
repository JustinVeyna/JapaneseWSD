'''
Created on Feb 27, 2018

@author: Justin Veyna
'''
from CNN_data_generator import *#word_to_senses
from bag_of_words import WordBag

WORD_BAG_DATA_PATH = 'S:/workspace/WSD/data/word_bag_data'


def word_bag_merge_data(words_list, words_senses):
    ret = []
    for word in words_senses:
        for sense in word:
            ret.append(words_list + list(sense.astype(float)))
    return ret

def word_bag_make_sentence_array(word2vec_dic, synset_data, sentence):
    sentence_array = WordBag(word2vec_dic)
    words_senses = []
    correctness = []
    merged_data = []
    for w in range(len(sentence)):
        word = sentence[w]
        sentence_array.add_words(word["text"])
        word_senses, correctness_temp = word_to_senses(synset_data, word)
        words_senses.append(word_senses)
        correctness.extend(correctness_temp)
        merged_data = word_bag_merge_data(sentence_array.words_list,words_senses)
    return (merged_data,correctness)

if __name__ == '__main__':
    word2vec_dic = load_word2vec_dic()
    synset_data = load_synset_data()
    z = 0
    for f in os.listdir(DOC_DATA_DIR):#document
        z+=1
        if z > MAX_DOCS:
            break
        print(f)
        doc_words_dic = load_words(f)
        sentence_count = 0
        data_x = []
        data_y = []
        for para in range(len(doc_words_dic)):#paragraph
            p = doc_words_dic[para]
            for _ in range(len(doc_words_dic[para])):#sentence
                sentence_count += 1
                s = doc_words_dic[para][sentence_count]
                sent_arr, correctness = word_bag_make_sentence_array(word2vec_dic, synset_data, s)
                data_x.extend(sent_arr)
                data_y.extend(correctness)
        with open(WORD_BAG_DATA_PATH+f+".pkl", "wb") as f:
            dump((data_x,data_y), f)
    '''
    data_x = numpy.array(data_x)
    data_y = numpy.array(data_y)
    with open(WORD_BAG_DATA_PATH, "wb") as f:
        dump((data_x,data_y), f)
    print(data_x[0], data_y[0])
    '''