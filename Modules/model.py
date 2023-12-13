import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self,lookback, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        #define Conv1D
        self.conv1d = nn.Conv1d(lookback, self.input_dim, kernel_size=1, stride=2)

        #Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)

        self.sequential = nn.Sequential(
            # self.conv1d,
            self.init_linear,
            nn.ReLU(),
            self.lstm,
        )

    def init_hidden(self):
        
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):

        lstm_out, self.hidden = self.sequential(input)
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred

class GRU(nn.Module):
    def __init__(self,lookback, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        #define Conv1D
        self.conv1d = nn.Conv1d(lookback, self.input_dim, kernel_size=1, stride=2)

        #Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)
        
        # Define the LSTM layer
        self.lstm = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)

        self.sequential = nn.Sequential(
            # self.conv1d,
            self.init_linear,
            nn.ReLU(),
            self.lstm,
        )

    def init_hidden(self):
        
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.sequential(input)
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred

class ElmanRNN(nn.Module):
    def __init__(self,lookback, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(ElmanRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        #define Conv1D
        self.conv1d = nn.Conv1d(lookback, self.input_dim, kernel_size=1, stride=2)

        #Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)
        
        # Define the LSTM layer
        self.lstm = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)

        self.sequential = nn.Sequential(
            # self.conv1d,
            self.init_linear,
            nn.ReLU(),
            self.lstm,
        )

    def init_hidden(self):
        
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.sequential(input)
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred