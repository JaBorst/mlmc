from mlmc.metrics import AccuracyTreshold
from mlmc.representation import threshold_max, threshold_hard, threshold_mcut
import torch


def test_AccuracyTresholdId():
    d = AccuracyTreshold(lambda x: x)

    example1 = (torch.tensor([[0,1,0], [0,1,0]]), torch.tensor([[0,1,0], [0,0,1]]) )
    example2 = (torch.tensor([[0,1,0], [0,1,0]]), torch.tensor([[0,1,0], [0,1,0]]) )

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.75

def test_AccuracyTreshold():
    d = AccuracyTreshold(threshold_max)

    example1 = (torch.tensor([[0,0.5,0], [0.3,0.7,0.1]]), torch.tensor([[0,1,0], [0,0,1]]) )
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]), torch.tensor([[0,1,0], [0,1,0]]) )

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.75

    d = AccuracyTreshold(threshold_hard, args_dict={"tr":0.5})

    example1 = (torch.tensor([[0,0.51,0], [0.3,0.7,0.1]]), torch.tensor([[0,1,0], [0,0,1]]) )
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]), torch.tensor([[0,1,0], [0,1,0]]) )

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.5

    d = AccuracyTreshold(threshold_mcut)

    example1 = (torch.tensor([[0,0.7,0.001], [0.3,0.7,0.1]]), torch.tensor([[0,1,0], [0,0,1]]) )
    example2 = (torch.tensor([[0,0.51,0.5], [-0.7,-0.5,-0.9]]), torch.tensor([[0,1,0], [0,1,0]]) )

    d.update(example1)
    assert d.compute() == 0.5

    d.update(example2)
    assert d.compute() == 0.5


