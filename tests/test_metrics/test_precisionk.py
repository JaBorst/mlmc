from mlmc.metrics import AccuracyTreshold
from mlmc.representation import threshold_max, threshold_hard, threshold_mcut
from mlmc.data import SingleLabelDataset, MultiLabelDataset
import torch


def test_AccuracyTresholdSingleIdentity():
    d = AccuracyTreshold(lambda x: x,is_multilabel=False)

    example1 = (torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([2, 1]))
    example2 = (torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([1, 1]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.75

def test_AccuracyTresholdSingleThreshold():
    d = AccuracyTreshold(threshold_max,is_multilabel=False)

    example1 = (torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([2, 1]))
    example2 = (torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([1, 1]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.75

    d = AccuracyTreshold(threshold_hard, args_dict={"tr":0.5})

    example1 = (torch.tensor([[0,0.51,0], [0.3,0.7,0.1]]), torch.tensor([2, 1]))
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]),  torch.tensor([1, 1]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.5

    d = AccuracyTreshold(threshold_mcut,is_multilabel=False)

    example1 = (torch.tensor([[0,0.7,0.001], [0.3,0.7,0.1]]), torch.tensor([2, 1]))
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]),  torch.tensor([1, 1]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.5



def test_AccuracyTresholdMultiIdentity():
    d = AccuracyTreshold(lambda x: x, is_multilabel=True)

    example1 = (torch.tensor([[0, 1, 1], [0, 1, 0]]), torch.tensor([[0,1,1], [0,0,1]]))
    example2 = (torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([[0,1,0], [0,1,0]]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.75

def test_AccuracyTresholdMultiThreshold():
    d = AccuracyTreshold(threshold_max, is_multilabel=True)

    example1 = (torch.tensor([[0, 1, 1], [0, 1, 0]]), torch.tensor([[0,1,1], [0,1,0]]))
    example2 = (torch.tensor([[0, 1, 0], [0, 1, 0]]), torch.tensor([[0,1,0], [0,1,0]]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.75

    d = AccuracyTreshold(threshold_hard, args_dict={"tr":0.5}, is_multilabel=True)

    example1 = (torch.tensor([[0,0.51,0], [0.3,0.7,0.1]]), torch.tensor([[0,1,1], [0,1,0]]))
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]),  torch.tensor([[0,1,0], [0,1,0]]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.5

    d = AccuracyTreshold(threshold_mcut, is_multilabel=True)

    example1 = (torch.tensor([[0,0.7,0.001], [0.3,0.7,0.1]]), torch.tensor([[0,1,1], [0,1,0]]))
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]), torch.tensor([[0,1,0], [0,1,0]]))

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.5
