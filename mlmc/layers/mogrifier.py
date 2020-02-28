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
    def __init__(self, input_size,  hidden_size, mogrify_steps=2, learn_initial_states=False, batch_first=True):
        super(MogrifierLSTM, self).__init__()
        self.cell = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_first=batch_first

        self.h = torch.nn.Parameter(torch.zeros(1, self.hidden_size))
        self.c = torch.nn.Parameter(torch.zeros(1, self.hidden_size))
        self.h.requires_grad = learn_initial_states
        self.c.requires_grad = learn_initial_states

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


class MogLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, mog_iterations: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iterations
        # Define/initialize all tensors
        self.Wih = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = torch.nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = torch.nn.Parameter(torch.Tensor(hidden_sz * 4))
        # Mogrifiers
        self.Q = torch.nn.Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        self.Whh = nn.init.orthogonal_(self.Whh)

        self.bih.data[self.hidden_size:2 * self.hidden_size] = 1
        self.bhh.data[self.hidden_size:2 * self.hidden_size] = 1

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor, init_states = None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt, ht)  # mogrification
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ### The LSTM Cell!
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            # outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (ht, Ct)
