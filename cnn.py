'''
Created on Feb 21, 2018

@author: Justin Veyna
'''

import torch.nn as nn
import numpy as np


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        """
        split into:
          sentence vectors
          sense vector
        linear the sentence vectors into a weight vector
        dot product with the sense vector
        out the sigged product of the dot product 
        """
        sentence_vectors = x[:-1]
        sense_vector = x[-1]
        out = self.linear(sentence_vectors)
        out = np.dot(out, sense_vector)
        out = self.sig(out)
        return out
    
if __name__ == '__main__':
    pass