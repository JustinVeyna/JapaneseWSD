'''
Created on Feb 28, 2018

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
from bag_of_words_data_generator import WORD_BAG_DATA_PATH, DOC_DATA_DIR
from bag_of_words import LinearRegression
import torch

if __name__ == '__main__':
 
    words_dict, _= load(open(WORD2VEC_DATA_DIR, 'rb'), encoding="bytes")
    
    input_size = len(words_dict)
    output_size = 1
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
        with open(WORD_BAG_DATA_PATH+f+".pkl", "rb") as fi:
            print(WORD_BAG_DATA_PATH+f+".pkl", "rb")
            x_train, y_train = load(fi)
        inputs = [Variable(FloatTensor(x)) for x in x_train][0]
        print(inputs.size())
        targets = Variable(FloatTensor(y_train))
    
        # Forward + Backward + Optimize
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print ('Epoch [%d/%d], Loss: %.4f'%(epoch+1, num_epochs, loss.data[0]))
            
    # Plot the graph
    #predicted = model(Variable(x_train)).data.numpy()
    #plt.plot(x_train, y_train, 'ro', label='Original data')
    #plt.plot(x_train, predicted, label='Fitted line')
    #plt.legend()
    #plt.show()
    
    # Save the Model
    torch.save(model.state_dict(), 'model.pkl')
    
    