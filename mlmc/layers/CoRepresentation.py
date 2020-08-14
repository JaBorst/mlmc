import torch

class ExpertCoRep(torch.nn.Module):
    """ Implementation of the expert system as propsed in http://proceedings.mlr.press/v97/zhang19l/zhang19l.pdf"""

    def __init__(self, in_features, out_features, centers,activation=torch.relu,  K=10, trainable=False):
        super(ExpertCoRep,self).__init__()
        self.K = K
        self.activation = activation
        self.projections = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=in_features, out_features=out_features)
             for _ in range(K)]
        )
        self.output_projection = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=out_features, out_features=out_features)
             for _ in range(K)]
        )
        assert len(centers.shape) == 1 or centers.shape[0]==K,"Centers must either be a tensor of single dimension or " \
                                                              "the first dimension must equal the number of experts in K"
        if len(centers.shape) == 1:
            self.single_center=True
            self.centers = torch.nn.Parameter(centers[None])
        elif centers.shape[0]==K:
            self.single_center=False
            self.centers= torch.nn.Parameter(centers)

        self.centers.requires_grad=trainable


    def expert(self, x, i):
        if self.single_center:
            return self.activation(self.projections[i](x-self.centers))
        else:
            return self.activation(self.projections[i](x-self.centers[i]))

    def forward(self,x):
        return torch.stack([self.expert(x,i) for i in range(self.K)]).sum(0)




