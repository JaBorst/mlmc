import torch

class DistanceAbstract(torch.nn.Module):
    def __init__(self,  distances=["cosine", "p1"], y=None):
        super(DistanceAbstract, self).__init__()
        self.distances = distances
        if y is not None:
            self.set_y(y)

    def _cov(self, m):
        m_exp = torch.mean(m.transpose(-1,-2), dim=-1)
        x = m.transpose(-1,-2) - m_exp.unsqueeze(-1)
        if len(x.shape)==2:
            cov = x.mm(x.transpose(-1,-2)) / (x.size(1) - 1)
        else:
            cov = torch.bmm(x,(x.transpose(-1,-2))) / (x.size(1) - 1)
        return cov

    def _p1(self, x, y):
        return (x - y).norm(p=1, dim=-1) / x.shape[-1]

    def _p2(self, x, y):
        return (x - y).norm(p=2, dim=-1) / x.shape[-1]

    def _p3(self, x, y):
        return (x - y).norm(p=3, dim=-1) / x.shape[-1]

    def _p4(self, x, y):
        return (x - y).norm(p=4, dim=-1) / x.shape[-1]

    def _jsd(self, x, y):
        xsf = torch.softmax(x, -1)
        return (xsf * torch.log(xsf / torch.softmax(y, -1))).sum(-1)

    def _scalar(self, x, y):
        return (x * y).sum(-1)

    def forward(self,x, y=None):

        xtmp, ytmp = self._reshape(x, self.y if y is None else y)
        if y is not None and "mahalanobis" in self.distances:
            self.cov = self._cov(y)
        return torch.stack([getattr(self,"_"+s)(xtmp,ytmp) for s in self.distances],-1)

    def set_y(self, y):
        self.y = y
        if "mahalanobis" in self.distances and y is not None:
            self.cov = self._cov(y)




#
#
# class DistanceAll(DistanceAbstract):
#     """ Implementation of the expert system as propsed in http://proceedings.mlr.press/v97/zhang19l/zhang19l.pdf"""
#
#     def __init__(self, **kwargs):
#         super(DistanceAll,self).__init__(**kwargs)
#
#     def _cosine(self, x, y):
#         if self.batch_broadcast:
#             return torch.matmul((x[:,0]/x[:,0].norm(dim=-1, p=2, keepdim=True)), (y[0,:,0]/y[0,:,0].norm(dim=-1, p=2, keepdim=True)).t()).transpose(1,2)
#         else:
#             return torch.bmm(x[:,0].squeeze(-1)/x[:,0].squeeze(-1).norm(dim=-1, p=2, keepdim=True), (y[:,:,0].squeeze(-1)/y[:,:,0].squeeze(-1).norm(dim=-1, p=2, keepdim=True)).transpose(1,2)).transpose(1,2)
#
#
#     def _mahalanobis(self,x,y):
#         diff = (x-y)
#         if len(self.cov.shape) == 2:
#             return torch.sqrt((torch.matmul(diff, self.cov.transpose(-1,-2)) *  diff).sum(-1)).squeeze(-1)
#         elif len(self.cov.shape) == 3:
#             return torch.sqrt((torch.einsum("ijkl,iml->ijkm",diff, self.cov.transpose(-1,-2)) *  diff).sum(-1)).squeeze(-1)
#
#     def _lrd(self, x , y):
#         dist1 = self._p2(x,y)
#         if self.batch_broadcast:
#             dist2 = self._p2(y[0],y[:,:,0])
#         else:
#             dist2 = self._p2(y,y[:,None][:,:,:,0])
#
#         dist2 = (dist2 + dist2.max()*torch.eye(dist2.shape[-1],dist2.shape[-1])).min(-1)[0]
#         return torch.sqrt((1-dist1/dist2.unsqueeze(-1))**2)
#
#     def _reshape(self, x,y):
#         assert not ( y.shape[0] == 1 and x.shape != 1), "This is ambigous. [y].shape[0] cant be 1"
#
#         if len(x.shape) == len(y.shape):
#             ytmp = y
#             self.batch_broadcast = False
#         else:
#             ytmp = y[None]
#             self.batch_broadcast = True
#
#         xtmp = x[:, None]
#         ytmp = ytmp[:, :, None]
#         # print(xtmp.shape, ytmp.shape)
#         return xtmp, ytmp

class DistanceCorrespondence(DistanceAbstract):
    """implementation of various comparison metrics"""
    def __init__(self, **kwargs):
        super(DistanceCorrespondence,self).__init__(**kwargs)

    def _cosine(self, x, y):
        r = (x / x.norm(p=2, dim=-1, keepdim=True) * (y / y.norm(p=2, dim=-1, keepdim=True))[None]).sum(-1)
        return r

    def _p1(self, x, y):
        return -(x - y[None]).norm(p=1, dim=-1)

    def _p2(self, x, y):
        return -(x - y[None]).norm(p=2, dim=-1)

    def _p3(self, x, y):
        return -(x - y[None]).norm(p=3, dim=-1)

    def _p4(self, x, y):
        return -(x - y[None]).norm(p=4, dim=-1)

    def _jsd(self, x, y):
        xsf = torch.softmax(x, -1)
        return 1-(xsf * torch.log(xsf / torch.softmax(y, -1)[ None])).sum(-1)

    def _scalar(self, x, y):
        return (x* y[None]).sum(-1)

    def _mahalanobis(self, x, y):
        diff = (x- y[None])
        return torch.sqrt((torch.matmul(diff, self.cov.transpose(-1, -2)) * diff).sum(-1)).squeeze(-1)

    def _lrd(self, x, y):
        dist1 = self._p2(x, y)
        dist2 = (y[None] - y[:, None]).norm(p=2, dim=-1)
        dist2 = (dist2 + dist2.max() * torch.eye(dist2.shape[-1], dist2.shape[-1]).to(x.device)).min(-1)[0]
        return torch.sqrt((1 - dist1 / dist2[None]) ** 2)

    def _reshape (self, x, y):
        assert x.shape[1] == y.shape[0], "Number of dimensions differ in x and y"
        return x,y


class Distance(DistanceAbstract):
    def __init__(self, **kwargs):
        super(Distance,self).__init__(**kwargs)

    def _cosine(self, x, y):
        r = torch.matmul(x/x.norm(p=2, dim=-1, keepdim=True),(y/ y.norm(p=2, dim=-1, keepdim=True)).t())
        return r

    def _p1(self, x, y):
        return (x[:,:,None]- y[None]).norm(p=1, dim=-1)

    def _p2(self, x, y):
        return (x[:,:,None]- y[None]).norm(p=2, dim=-1)

    def _p3(self, x, y):
        return (x[:,:,None]- y[None]).norm(p=3, dim=-1)

    def _p4(self, x, y):
        return (x[:,:,None] - y[None]).norm(p=4, dim=-1)

    def _jsd(self, x, y):
        xsf = torch.softmax(x, -1)
        return (xsf [:,:,None] * torch.log(xsf[:,:,None] / torch.softmax(y, -1)[None,None])).sum(-1)

    def _scalar(self, x, y):
        return torch.matmul(x, y.t())

    def _mahalanobis(self, x, y):
        diff = (x[:,:,None] - y[None,None])
        return torch.sqrt((torch.matmul(diff, self.cov.transpose(-1, -2)) * diff).sum(-1)).squeeze(-1)

    def _min_distances(self, y):
        if y.shape[0] > 1000:
            batch_size = 100
            dis = []
            for b in range(0, y.shape[0], batch_size):
                batch = y[b:(b + batch_size)]
                batch_distances = (batch[None] - y[:, None]).norm(p=2, dim=-1)
                # m = batch_distances.max()
                # for i in range(batch_distances.shape[-1]):
                #     batch_distances[i, i] = m
                batch_distances[batch_distances==0] = batch_distances.max()
                batch_distances_min = batch_distances.min(0)[0]
                dis.append(batch_distances_min)
            return torch.cat(dis, 0)
        else:
            dist2 = (y[None] - y[:, None]).norm(p=2, dim=-1)
            dist2 = (dist2 + dist2.max() * torch.eye(dist2.shape[-1], dist2.shape[-1]).to(dist2.device)).min(-1)[0]
            return dist2

    def _lrd(self, x, y):
        dist1 = self._p2(x, y)
        return torch.sqrt((1 - dist1 / self.min_distances[None,None]) ** 2)

    def _reshape(self, x,y):
        assert len(y.shape)==2, "y has to be 2D"
        return x, y

    def forward(self,x, y=None):
        if y is not None:
            self.set_y(y)
        xtmp, ytmp = self._reshape(x, self.y)
        return torch.stack([getattr(self,"_"+s)(xtmp,ytmp) for s in self.distances],-1)

    def set_y(self, y):
        self.y = y
        if "mahalanobis" in self.distances and y is not None:
            self.cov = self._cov(y)
        if "lrd" in self.distances:
            self.min_distances = self._min_distances(y)




class Similarity(DistanceAbstract):
    def __init__(self, fuzzyness=1.0, **kwargs):
        super(Similarity,self).__init__(**kwargs)
        self.fuzzyness = fuzzyness

    def _cosine(self, x, y):
        r = torch.matmul(x/x.norm(p=2, dim=-1, keepdim=True),(y/ y.norm(p=2, dim=-1, keepdim=True)).t())
        return r

    def _p1(self, x, y):
        return torch.exp(-self.fuzzyness*(x[:,:,None]- y[None]).norm(p=1, dim=-1))

    def _p2(self, x, y):
        return torch.exp(-self.fuzzyness*(x[:,:,None]- y[None]).norm(p=2, dim=-1))

    def _p3(self, x, y):
        return torch.exp(-self.fuzzyness*(x[:,:,None]- y[None]).norm(p=3, dim=-1))

    def _p4(self, x, y):
        return torch.exp(-self.fuzzyness*(x[:,:,None] - y[None]).norm(p=4, dim=-1))

    def _jsd(self, x, y):
        xsf = torch.softmax(x, -1)
        return torch.exp(-self.fuzzyness* (xsf [:,:,None] * torch.log(xsf[:,:,None] / torch.softmax(y, -1)[None,None])).sum(-1))

    def _scalar(self, x, y):
        return torch.softmax(torch.matmul(x, y.t()),-2)

    def _mahalanobis(self, x, y):
        diff = (x[:,:,None] - y[None,None])
        return torch.exp(-self.fuzzyness*torch.sqrt((torch.matmul(diff, self.cov.transpose(-1, -2)) * diff).sum(-1)).squeeze(-1))

    def _min_distances(self, y):
        if y.shape[0] > 1000:
            batch_size = 100
            dis = []
            for b in range(0, y.shape[0], batch_size):
                batch = y[b:(b + batch_size)]
                batch_distances = (batch[None] - y[:, None]).norm(p=2, dim=-1)
                # m = batch_distances.max()
                # for i in range(batch_distances.shape[-1]):
                #     batch_distances[i, i] = m
                batch_distances[batch_distances==0] = batch_distances.max()
                batch_distances_min = batch_distances.min(0)[0]
                dis.append(batch_distances_min)
            return torch.cat(dis, 0)
        else:
            dist2 = (y[None] - y[:, None]).norm(p=2, dim=-1)
            dist2 = (dist2 + dist2.max() * torch.eye(dist2.shape[-1], dist2.shape[-1]).to(dist2.device)).min(-1)[0]
            return dist2

    def _lrd(self, x, y):
        dist1 = self._p2(x, y)
        return torch.exp(-self.fuzzyness*(1 - dist1 / self.min_distances[None,None]) ** 2)

    def _reshape(self, x,y):
        assert len(y.shape)==2, "y has to be 2D"
        return x, y

    def forward(self,x, y=None):
        if y is not None:
            self.set_y(y)
        xtmp, ytmp = self._reshape(x, self.y)
        return torch.stack([getattr(self,"_"+s)(xtmp,ytmp) for s in self.distances],-1)

    def set_y(self, y):
        self.y = y
        if "mahalanobis" in self.distances and y is not None:
            self.cov = self._cov(y)
        if "lrd" in self.distances:
            self.min_distances = self._min_distances(y)



# y = torch.rand(1000, 300)
# d = Distance(distances=["lrd","p1", "p2", "p3", "cosine",  "mahalanobis", "scalar", "jsd"])
# # d.set_y(x)
#
# x = torch.rand(2,140, 300)
# y = torch.rand(2000, 300)
# d.set_y(y)
# print(d(x).shape)
# #
# # x = torch.rand(1,140, 300)
# # y = torch.rand(140, 300)
# # d(x,y).shape
# #
# # # x = torch.rand(2,200, 300)
# # # # print(d(x,y).shape)
# #
# y = torch.rand(140, 300)
# #
#
#
#
