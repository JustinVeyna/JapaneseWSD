'''
Created on Feb 27, 2018

@author: Justin Veyna
'''
from pickle import load
from sense_guesser import WORD2VEC_DATA_DIR

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
from torch import from_numpy
from pickle import load
import os
#from bag_of_words_data_generator import WORD_BAG_DATA_PATH, DOC_DATA_DIR

class WordBag():
    '''
    classdocs
    '''
    TRUE_VALUE = 1
    FALSE_VALUE = 0
    def __init__(self, words_dict):
        '''
        Constructor
        '''
        self.words_dict = dict(list(zip(words_dict.keys(), list(range(len(words_dict.keys()))))))
        self.words_list = [WordBag.FALSE_VALUE for _ in range(len(words_dict))]
    
    def add_words(self, words):
        if not words == None:
            if type(words) == str:
                self._set_true(words)
            else:#type==list
                for word in words:
                    self.add_words(self, word)
    
    def _set_true(self, word):
        if word in self.words_dict:
            word_index = self.words_dict[word]
            self.words_list[word_index] = WordBag.TRUE_VALUE




class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size-output_size, output_size)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        """
        split x into:
          bag_of_words
          sense vector
        linear the sentence vectors into a weight vector
        dot product with the sense vector
        out the sigged product of the dot product 
        """
        bag_of_words = x[:-63]
        sense_vector = x[-64:]
        out = self.linear(bag_of_words)
        out = np.dot(out, sense_vector)
        out = self.sig(out)
        return out
    
if __name__ == '__main__':
    
    words_dict, _= load(open(WORD2VEC_DATA_DIR, 'rb'), encoding="bytes")
    
    input_size = len(words_dict)
    output_size = 64
    learning_rate = .001
    num_epochs = 0
    model = LinearRegression(input_size, output_size)
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    epoch = 0
    for f in os.listdir(DOC_DATA_DIR):#document
        if epoch >= num_epochs and num_epochs != 0:
            break
        else:
            epoch += 1
        with open(WORD_BAG_DATA_PATH+f+".pkl", "wb") as f:
            x_train, y_train = load(f)
    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        targets = Variable(y_train)
    
        # Forward + Backward + Optimize
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print ('Epoch [%d/%d], Loss: %.4f'%(epoch+1, num_epochs, loss.data[0]))
            
    # Plot the graph
    predicted = model(Variable(x_train)).data.numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()
    
    # Save the Model
    torch.save(model.state_dict(), 'model.pkl')
    