'''
Created on Jan 23, 2018

@author: Justin Veyna

References:
    https://docs.python.org/3.6/library/xml.etree.elementtree.html
'''

import xml.etree.ElementTree as ET
from collections import defaultdict
import pickle
import os

BASE_DIR = 'S:/workspace/WSD/'
DATA_DIR = BASE_DIR + 'jsemcor-2012-01/'
SAVE_DIR = BASE_DIR + "jsemcor-2012-01-pickled/"
SAVE_DIR_ENDING = ".pkl"

DEFAULT_FILE_PATH = "jsemcor-2012-01/br-a01.xml"


class XMLDoc():
    '''
    A wrapper class for the XML documents.
    '''
    def __init__(self, xml_file = "jsemcor-2012-01/br-a01.xml", debug = False):
        '''
        xml_file = path of xml file to open. ie:"jsemcor-2012-01/br-a01.xml" 
        '''
        self.debug = debug
        
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        
        if self.debug:
            print("Initialized")
            print(self.tree)
            print(self.root)
    
    def get_terms(self):
        return self.root.iter("term")
    def get_words(self):
        return self.root.iter("wf")
    def get_word_sense_by_id(self, wid):
        sense = self.root.find(".//*[@id=\'"+wid+"\']/../../externalReferences/externalRef/externalRef")
        if sense != None:
            SKIP = len("jpn-11-")
            sense = sense.attrib["reference"][SKIP:]
        return sense


def ddict():
    return defaultdict(dict)
def dddict():
    return defaultdict(ddict)
def ddddict():
    return defaultdict(dddict)
    
def get_sense_linked_terms(xml_file = DEFAULT_FILE_PATH):
    slt = ddddict()
    docu = XMLDoc(xml_file)
    for word in docu.get_words():
        #print(word.attrib)
        wid = word.attrib["wid"]
        split_wid = wid[1:].split(".")
        p, s, w = map(int, split_wid)
        slt[p][s][w]["wid"] = wid
        #print(word.text)
        slt[p][s][w]["text"] = word.text
        ref = docu.get_word_sense_by_id(wid)
        #print(ref)
        slt[p][s][w]["sense"] = ref
    return slt

def original_doc(xml_file = DEFAULT_FILE_PATH, t=list):
    '''
    Prints the original document without any of the XML tags or labels
    t = return type (list or string)
    '''
    orig = t()
    docu = XMLDoc(xml_file)
    for term in docu.get_words():
        if term.text != None:
            orig += t(term.text)
    return orig

def print_original_doc(xml_file = DEFAULT_FILE_PATH):
    '''Prints the original document without any of the XML tags or labels'''
    print(original_doc(xml_file, t=str))

if __name__ == "__main__":
    for f in os.listdir(DATA_DIR):
        print(f)
        words = get_sense_linked_terms(DATA_DIR+f)
        #print(words)
        with open(SAVE_DIR + f + SAVE_DIR_ENDING, "wb") as s:
            pickle.dump(words,s)
        