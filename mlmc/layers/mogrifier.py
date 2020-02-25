import torch
import torch.nn as nn
import math


class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.q = nn.Linear(hidden_size, input_size)
        self.r = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.init_parameters()
        self.mogrify_steps = mogrify_steps

    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-std, std)

    def mogrify(self, x, h):
        for i in range(1, self.mogrify_steps + 1):
            if i % 2 == 0:
                h = (2 * torch.sigmoid(self.r(x))) * h
            else:
                x = (2 * torch.sigmoid(self.q(h))) * x
        return x, h

    def forward(self, x, states):
        """
        inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)
        """
        ht, ct = states
        x, ht = self.mogrify(x, ht)  # Note: This should be called every timestep
        gates = self.x2h(x) + self.h2h(ht)  # (batch_size, 4 * hidden_size)
        in_gate, forget_gate, new_memory, out_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        new_memory = self.tanh(new_memory)
        c_new = (forget_gate * ct) + (in_gate * new_memory)
        h_new = out_gate * self.tanh(c_new)

        return h_new, c_new

class MogrifierLSTM(torch.nn.Module):
    def __init__(self, input_size,  hidden_size, mogrify_steps=2, learn_initial_states=True, batch_first=True):
        super(MogrifierLSTM, self).__init__()
        self.cell = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)
        self.h = torch.nn.Parameter(torch.zeros(1, hidden_size))
        self.c = torch.nn.Parameter(torch.zeros(1, hidden_size))
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.h.requires_grad = learn_initial_states
        self.c.requires_grad = learn_initial_states
        self.batch_first=batch_first

    def forward(self, x):
        if not self.batch_first:
            x = x.permute(1,0,2)

        max_len=x.shape[1]
        batch_size=x.shape[0]
        hidden_states = []
        h = self.h.repeat(batch_size,1)
        c = self.c.repeat(batch_size,1)

        for step in range(max_len):
            t = x[:, step]
            h, c = self.cell(t, (h, c))
            hidden_states.append(h.unsqueeze(1))

        return torch.cat(hidden_states, dim = 1), (h, c)
