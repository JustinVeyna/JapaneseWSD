'''
Created on Feb 23, 2018

@author: Justin Veyna
'''
from abstract_record_loader import AbstractRecordLoader
from collections import namedtuple


class SynsetDefLoader(AbstractRecordLoader):
    def __init__(self):
        super().__init__()
        self.syndef = namedtuple('SynDef', 'synset lang defin sid')

    def load_syndef_with_sense(self, sense, lang="eng"):
        cur = self.conn.execute("select * from synset_def where synset=? and lang=?",
                               (sense.synset, lang))
        return [self.syndef(*row) for row in cur]

