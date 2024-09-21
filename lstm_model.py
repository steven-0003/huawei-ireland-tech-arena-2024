import torch
import torch.nn as nn 



class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_stacked_layers, lookahead):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, lookahead)



    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to("cpu")
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to("cpu")

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

