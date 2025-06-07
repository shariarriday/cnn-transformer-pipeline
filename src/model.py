import torch.nn as nn
import torch

# Define our network class using nn.Module
class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Define layers for the MLP block
        self.norm1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.norm2 = nn.LayerNorm(input_size//2)
        self.fc2 = nn.Linear(input_size//2, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.act = nn.ELU()

    def forward(self, x):
        # Forward pass through the MLP block
        x = self.act(self.norm1(x))
        skip = x  # Skip connection
        x = self.act(self.norm2(self.fc1(x)))
        x = self.fc2(x)
        return x + skip

# Define the LSTM-based network
class LSTM(nn.Module):
    def __init__(self, output_size, num_blocks=1, num_layers=64, input_size=99, hidden_size=128):
        super(LSTM, self).__init__()
        # Define layers for input MLP, LSTM, residual blocks, and output linear layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=.5, bidirectional=True)
        blocks = [ResBlockMLP(hidden_size * 2, hidden_size * 2) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(.5)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, input_seq, h0, c0):
        # Pass the input MLP output through the LSTM block
        output, (hidden_out, mem_out) = self.lstm(input_seq, (h0, c0))
       
        # Pass the LSTM output through residual blocks
        x = self.act(self.res_blocks(output))
        x = self.dropout(x)
        x = self.fc_out(x)
       
        # Pass the output of the residual blocks through the final linear layer
        return x, hidden_out, mem_out
