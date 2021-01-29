import torch.nn as nn
import torch.nn.functional as nf


# Model
class BasicNeuralNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        """ Basic structure of a 1-layer neural network of a sigmoid output

        :param input_dim: Input number of features
        :param hidden_dim: Hidden layer number of features
        :param output_dim: Output dimension
        """
        super(BasicNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(0.15)
        self.relu = nf.relu
        self.sig = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_x):
        """ Feedforward behavior of the net

        :param input_x: A batch of input features
        :return: A single value of output
        """
        out = self.relu(self.fc1(input_x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sig(out)  # or self.log_softmax(out)


