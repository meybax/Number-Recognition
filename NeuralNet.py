import torch.nn as nn
import torch.nn.functional as f


# neural network class with constructor and forward function
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = f.relu(self.fc1(x))
        out = f.relu(self.fc2(out))
        out = self.fc3(out)
        return out
