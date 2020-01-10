import torch
import networkx

adjacency = torch.Tensor([[0,1,0,0],
                          [1,0,1,0],
                          [0,0,0,1],
                          [1,0,0,0]])
adjacency = adjacency/torch.norm(adjacency,p=1, dim=-1).unsqueeze(-1)
propability = torch.Tensor([100,0,0,0])
propability = torch.matmul(adjacency.transpose(0,1), propability)