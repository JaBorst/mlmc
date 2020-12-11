from torch.utils.data import Dataset
import numpy as np


class GraphDataset(Dataset):
    def __init__(self, graph, n=2, target=["description", "extract", "label"], **kwargs):
        super(GraphDataset, self).__init__(**kwargs)
        self.graph = graph.to_undirected(reciprocal=False)
        self.nodes = list(graph.nodes)
        self.n = n
        self.d = dict(self.graph.nodes(True))
        self.target=target

    def __getitem__(self, idx):
        # print(idx)
        neighbours = list(self.graph.neighbors(self.nodes[idx]))
        searching = True
        while (searching):
            neg_idx = np.random.randint(len(self.nodes))
            if (self.nodes[neg_idx], self.nodes[idx]) not in self.graph.edges:
                searching = False
        negative_neighbours = list(self.graph.neighbors(self.nodes[neg_idx]))

        random_neighbours =  np.random.choice(neighbours, self.n,
                                               replace=False if self.n < len(neighbours) else True).tolist()
        random_negative_neighbours = [self.nodes[neg_idx]] + np.random.choice(negative_neighbours, self.n,
                                                                      replace=False if self.n < len(
                                                                          negative_neighbours) else True).tolist()

        return {'input': self._extract(self.nodes[idx]),
                'neighbours':[self._extract(x) for x in random_neighbours],
                'negatives': [self._extract(x) for x in random_negative_neighbours]}

    def _extract(self, x):
        r = x
        if self.target == ["label"]:
            return x
        for t in self.target:
            if t in self.d[x].keys():
                r += ": " + self.d[x][t]
                break
        return r


    def __len__(self):
        return len(self.nodes)



from copy import deepcopy
from ...data import get_singlelabel_dataset, get_multilabel_dataset

class ZeroshotDataset:
    def __init__(self, dataset, zeroshot_classes=None):
        if isinstance(dataset,str):
            try:
                dataset = get_singlelabel_dataset("rcv1")
            except AssertionError:
                dataset = get_multilabel_dataset("rcv1")

        train = dataset.get("train", None)
        valid = dataset.get("valid", None)
        test = dataset.get("test", None)

        self.zeroshot_classes = zeroshot_classes

        self.zeroshot_data = {}

        if train is not None:
            data = deepcopy(train)
            data.remove(zeroshot_classes)
            self.zeroshot_data["train"] = data

        if valid is not None:
            gzsl_data = deepcopy(valid)
            gzsl_data.classes = valid.classes
            self.zeroshot_data["valid_gzsl"] = gzsl_data

            zsl_data = deepcopy(valid)
            zsl_data.reduce(dict(zip(self.zeroshot_classes, range(len(zeroshot_classes)))))
            self.zeroshot_data["valid_zsl"] = zsl_data

            nsl_data = deepcopy(valid)
            nsl_data.reduce(self.zeroshot_data["train"].classes)
            self.zeroshot_data["valid_nsl"] = nsl_data

        if test is not None:
            gzsl_data = deepcopy(test)
            gzsl_data.classes = test.classes
            self.zeroshot_data["test_gzsl"] = gzsl_data

            zsl_data = deepcopy(test)
            zsl_data.reduce(dict(zip(self.zeroshot_classes, range(len(zeroshot_classes)))))
            self.zeroshot_data["test_zsl"] = zsl_data

            nsl_data = deepcopy(test)
            nsl_data.reduce(self.zeroshot_data["train"].classes)
            self.zeroshot_data["test_nsl"] = nsl_data

    def get(self, n):
        return self.zeroshot_data[n]

    def __getitem__(self, item):
        return self.zeroshot_data[item]

