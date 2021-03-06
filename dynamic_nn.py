import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import *


class DynamicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim, output_type):
        """ Defines layers of a neural network.

        :param input_dim: Number of input features
        :param hidden_dim_list: List of hidden layer dimensions, [3, 4, 4]
        :param output_dim: Number of output
        :param output_type: 'continuous' or 'binary'
        """
        assert len(hidden_dim_list) > 0 and all([isinstance(i, int) for i in hidden_dim_list])
        super(DynamicNet, self).__init__()
        self.output_type = output_type
        full_dim_list = [input_dim] + hidden_dim_list + [output_dim]
        self.full_dim_list = full_dim_list
        self.fc_dict = {}
        for i in range(len(full_dim_list) - 1):
            fc = nn.Linear(full_dim_list[i], full_dim_list[i + 1])
            setattr(self, f'fc{i + 1}', fc)
            self.fc_dict[i + 1] = f'fc{i + 1}'
        self.dropout = nn.Dropout(0.15)
        self.sig = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)
        log_info('DynamicNet created')

    def forward(self, x):
        """ Feedforward behavior of the net

        :param x: A batch of input features
        :return: A single value of output
        """
        for i in range(len(self.fc_dict)):
            fc = getattr(self, self.fc_dict[i + 1])
            x = fc(x)
            if i + 1 != len(self.fc_dict):
                if self.output_type == 'binary':
                    x = F.relu(x)  # activation on hidden layers
                else:
                    x = torch.tanh(x)
                x = self.dropout(x)
        if self.output_type == 'binary':
            x = self.sig(x)  # Sigmoid function on output if output type is a binary label
        return x
