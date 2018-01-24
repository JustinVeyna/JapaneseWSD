'''
Created on Jan 23, 2018

@author: Justin Veyna

References:
    https://docs.python.org/3.6/library/xml.etree.elementtree.html
'''

import xml.etree.ElementTree as ET

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
    print_original_doc()