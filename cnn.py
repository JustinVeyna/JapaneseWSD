'''
Created on Feb 21, 2018

@author: Justin Veyna
'''


import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
from torch import from_numpy
from pickle import load
CNN_DATA_PATH = 'S:/workspace/WSD/data/cnn_data.pkl'


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
    with open(CNN_DATA_PATH, "rb") as f:
        x_train, y_train = load(f)
    print(len(x_train))
    print(from_numpy(x_train.astype(float)))
    
    input_size = LongTensor([None,64])
    output_size = 1
    learning_rate = .001
    num_epochs = 100000
    model = LinearRegression(input_size, output_size)
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        inputs = Variable(LongTensor(x_train))
        targets = Variable(LongTensor(y_train))
    
        # Forward + Backward + Optimize
        optimizer.zero_grad()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print ('Epoch [%d/%d], Loss: %.4f'%(epoch+1, num_epochs, loss.data[0]))
            
    # Plot the graph
    predicted = model(Variable(LongTensor(x_train))).data.numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.legend()
    plt.show()
    
    # Save the Model
    torch.save(model.state_dict(), 'model.pkl')
    
    
    
    
    
    
    
    
    
    
    
    
    
    