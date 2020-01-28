# # import torch
# #
# # class Prob(torch.nn.Module):
# #     def __init__(self, n_labels):
# #         super(Prob,self).__init__()
# #         self.corr = torch.nn.Parameter(torch.randn(n_labels, n_labels))
# #     def forward(self, x):
# #         x = (torch.unsqueeze(x, -1) * self.corr).sum(-2)
# #         return x
# #
# #
#
# import numpy as np
# import torch
#
#
# class LSTMRD(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, bias=True,
#                  batch_first=False, dropout=0.5, recurrent_dropout=0.5, bidirectional=False):
#         super(LSTMRD,self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.batch_first = batch_first
#         self.dropout = torch.nn.Dropout(dropout)
#         self.recurrent_dropout = torch.nn.Dropout(recurrent_dropout)
#         self.bidirectional = bidirectional
#
#         self.forward_cell = torch.nn.LSTMCell(self.input_size,self.hidden_size,bias=self.bias)
#
#         # self.xo = torch.nn.Linear(in_features=input_size, out_features=self.hidden_size)
#         # self.ho = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
#
#         if bidirectional:
#             self.backward_cell  = torch.nn.LSTMCell(self.input_size,self.hidden_size,bias=self.bias)
#             # self.bxo = torch.nn.Linear(in_features=input_size, out_features=self.hidden_size)
#             # self.bho = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
#
#
#
#     def forward(self, x, hidden=None):
#         if self.batch_first:
#             x = x.permute(1,0,2)
#         if self.dropout.p>0:
#             x = self.dropout(x)
#
#         output = []
#         hx = hidden[0][0]
#         cx = hidden[1][0]
#
#         rec_dp_mask = self.recurrent_dropout(torch.ones_like(hx))
#
#         for i in range(x.shape[0]):
#             hx, cx = self.forward_cell(x[i], (hx, cx))
#             output.append(hx)
#             if self.recurrent_dropout.p >0:
#                 cx = cx*rec_dp_mask
#         forward = torch.stack(output,dim=0)
#
#         if self.bidirectional:
#             x = x.flip(0)
#             hx = hidden[0][1]
#             cx = hidden[1][1]
#             output = []
#             for i in range(x.shape[0]):
#                 hx, cx = self.backward_cell(x[i], (hx, cx))
#                 output.append(hx)
#                 if self.recurrent_dropout.p>0:
#                     cx = cx * rec_dp_mask
#
#             output = torch.stack(output, dim=0)
#             forward = torch.cat([forward, output.flip(0)],dim=-1)
#
#         if self.batch_first:
#             forward = forward.permute(1,0,2)
#         return forward
#
#
#
# l = LSTMRD(30,55,bias=True,batch_first=True,bidirectional=True)
# l(torch.rand((16,140,30)), (torch.rand((2,16,55)),torch.rand((2,16,55)))).shape

import mlmc
mlmc.data.export(*mlmc.data.load_20newsgroup(), "/home/jborst/20newsgroup")
mlmc.data.export(*mlmc.data.load_blurbgenrecollection(), "/home/jborst/blurbgenrecollection")