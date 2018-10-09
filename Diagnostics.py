import pandas as pd
import torch
import torch.nn as nn
import NeuralNet

# Note: not actually called in Main. Used to diagnose and optimize algorithm

# important variables
input_size = 784
hidden_size = 500
num_classes = 10
learning_rate = 0.003
iterations = 50

# loading and separating data sets
data = pd.read_csv('train.csv')
train = torch.tensor(data.iloc[:30000, :].values)
test = torch.tensor(data.iloc[30000:, :].values)

# instantiation of neural network
net = NeuralNet.NeuralNet(input_size, hidden_size, num_classes)
print(net)

# functions for training neural network
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# iterative process of training neural network
print('\nTraining network...')
for i in range(iterations):

    # computes the hypothesis and cost
    outputs = net(train[:, 1:].float())
    cost = criterion(outputs, train[:, 0])

    # edits the parameters
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print((i + 1), '- Cost: {:.4f}'.format(cost.item()))

# testing neural network on separate data set
print('\n\nTesting network...')
with torch.no_grad():

    correct = 0
    total = 0
    outputs = net(test[:, 1:].float())
    cost = criterion(outputs, test[:, 0])
    _, predicted = torch.max(outputs.data, 1)
    total = test[:, 0].size(0)
    correct = (predicted == test[:, 0]).sum().item()

    print('Accuracy: {} %'.format(100 * correct / total))
    print('Cost: {:.4f}'.format(cost.item()))
