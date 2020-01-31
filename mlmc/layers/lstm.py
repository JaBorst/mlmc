import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes, sorted_indices, unsorted_indices = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes, sorted_indices, unsorted_indices)
        else:
            return x


class LSTM(nn.LSTM):
    """A copy of the better LSTM github repository"""
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        # self.input_drop = torch.nn.Dropout(dropouti)
        # self.output_drop = torch.nn.Dropout(dropouto)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1
        self.flatten_parameters()
    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training)
        self.flatten_parameters()

    def forward(self, input, hx=None):
        self._drop_weights()
        seq, state = super().forward(input, hx=hx)
        return seq, state




class LSTMRD(torch.nn.Module):
    """
    A self implementation of the LSTM with a tensorflow like recurrent dropout on the activations of the
    reccurrent cell
    """
    def __init__(self, input_size, hidden_size, bias=True,
                 batch_first=False, dropout=0.5, recurrent_dropout=0.5, bidirectional=False):
        super(LSTMRD,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = torch.nn.Dropout(dropout)
        self.recurrent_dropout = torch.nn.Dropout(recurrent_dropout)
        self.bidirectional = bidirectional

        self.forward_cell = torch.nn.LSTMCell(self.input_size,self.hidden_size,bias=self.bias)

        if bidirectional:
            self.backward_cell  = torch.nn.LSTMCell(self.input_size,self.hidden_size,bias=self.bias)

    def forward(self, x, hidden=None):
        if self.batch_first:
            x = x.permute(1,0,2)
        if self.dropout.p>0:
            x = self.dropout(x)
        output = []
        hx = hidden[0][0]
        cx = hidden[1][0]
        for i in range(x.shape[0]):
            hx, cx = self.forward_cell(x[i], (hx, cx))
            if self.recurrent_dropout.p >0:
                cx = self.recurrent_dropout(cx)
            output.append(hx)
        forward = torch.stack(output,dim=0)
        if self.bidirectional:
            x = x.flip(0)
            hx = hidden[0][1]
            cx = hidden[1][1]
            output = []
            for i in range(x.shape[0]):
                hx, cx = self.backward_cell(x[i], (hx, cx))
                if self.recurrent_dropout.p>0:
                    cx = self.recurrent_dropout(cx)
                output.append(hx)
            output = torch.stack(output, dim=0)
            forward = torch.cat([forward, output.flip(0)],dim=-1)

        if self.batch_first:
            forward = forward.permute(1,0,2)
        return forward
