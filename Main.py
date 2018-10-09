import pandas as pd
import torch
import torch.nn as nn
import NeuralNet
import Training
import Interpreter

# important variables
input_size = 784
hidden_size = 500
num_classes = 10
learning_rate = 0.001
iterations = 50

# loading and separating data sets from csv file
data = pd.read_csv('train.csv')
train = torch.tensor(data.iloc[:, :].values)

# instantiation of neural network
net = NeuralNet.NeuralNet(input_size, hidden_size, num_classes)
print(net)

# functions for training neural network
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
x = train[:, 1:]
y = train[:, 0]

# iterative process of training neural network
print('\nTraining network...\n')
Training.train(net, x, y, optimizer, criterion, iterations)

# process of drawing and training neural network
print('\n\nInterpreting images...\n')
x = torch.zeros(0, 0)
y = torch.zeros(0, 0)
while True:
    # drawing and saving image
    print('\nPlease draw a digit')
    img = Interpreter.interpreter()
    output = net(img.float())
    _, prediction = torch.max(output.data, 0)
    print('Guess: {}'.format(prediction))
    
    # adding each drawing to the data set to continuously train the neural network
    x = torch.cat((x, torch.unsqueeze(img, 0).float()), 0)
    digit = torch.tensor(prediction)
    train = input('Correct? ')
    if train[0] == 'n':
        digit = torch.tensor(int(input('Actual digit? ')))
    y = torch.cat((y, torch.unsqueeze(digit, 0).float()), 0)
    Training.train(net, x, y.long(), optimizer, criterion, int(iterations / 5))

    cont = input('Try Again? ')
    if cont[0] == 'n':
        break
