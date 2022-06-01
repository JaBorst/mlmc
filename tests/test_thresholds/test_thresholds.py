import mlmc
import torch
from mlmc.thresholds import thresholds

def test_threshold_max():

    example1 = torch.tensor([[0.5, 0.6, 0.9, 0.2]])
    example2 = torch.tensor([[0.52, 0.53, 0.881, 0.882]])
    assert torch.equal(thresholds.threshold_max(example1), torch.tensor([[0., 0., 1., 0.]]))
    assert torch.equal(thresholds.threshold_max(example2), torch.tensor([[0., 0., 0., 1.]]))

def test_threshold_hard():

    example1 = torch.tensor([[0.1, 0.51, 0.49, 0.8]])
    example2 = torch.tensor([[0.9, 0.6, 0.7, 0.3]])
    assert torch.equal(thresholds.threshold_hard(example1), torch.tensor([[0, 1, 0, 1]], dtype = torch.int))
    assert torch.equal(thresholds.threshold_hard(example2, tr=0.8), torch.tensor([[1, 0, 0, 0]], dtype = torch.int))

def test_threshold_mean():

    example1 = torch.tensor([[0., 0.25, 0.5, 1.]])
    example2 = torch.tensor([[0.49, 0.51, 0.49, 0.51]])
    assert torch.equal(thresholds.threshold_mean(example1), torch.tensor([[0., 0., 1., 1.]]))
    assert torch.equal(thresholds.threshold_mean(example2), torch.tensor([[0., 1., 0., 1.]]))

def test_threshold_mcut():

    example1 = torch.tensor([[0.9, 0.8, 0.8, 0.7, 0.3, 0.2]])
    example2 = torch.tensor([[0.9, 0.7, 0.2, 0.2, 0.1]])
    example3 = torch.tensor([[0.8, 0.797, 0.795, 0.793, 0.79]])
    example4 = torch.tensor([[0.000001, 0.000001, 0.000002]])
    example5 = torch.tensor([[0.999999, 0.999999, 0.999998]])
    assert torch.equal(thresholds.threshold_mcut(example1), torch.tensor([[1., 1., 1., 1., 0., 0.]]))
    assert torch.equal(thresholds.threshold_mcut(example2), torch.tensor([[1., 1., 0., 0., 0.]]))
    assert torch.equal(thresholds.threshold_mcut(example3), torch.tensor([[1., 0., 0., 0., 0.]]))
    assert torch.equal(thresholds.threshold_mcut(example4), torch.tensor([[0., 0., 1.]]))
    assert torch.equal(thresholds.threshold_mcut(example5), torch.tensor([[1., 1., 0.]]))

def test_threshold_wrapper():
    example1 = torch.tensor([[0.38, 0.83, 0.21, 0.829]])
    example2 = torch.tensor([[0.3, 0.4, 0.5, 0.6]])
    example3 = torch.tensor([[0.75, 0.5, 0.2, 0.75]])
    example4 = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.4, 0.3]])
    example5 = torch.tensor([[0.8, 0.7, 0.2, 0.5, 0.1, 0.15]])
    ind = torch.tensor([0, 0, 1, 0, 1, 1])
    assert torch.equal(mlmc.thresholds.get(name="max").__call__(example1), torch.tensor([[0., 1., 0., 0.]]))
    assert torch.equal(mlmc.thresholds.get(name="hard").__call__(example2), torch.tensor([[0, 0, 0, 1]], dtype = torch.int))
    assert torch.equal(mlmc.thresholds.get(name="mean").__call__(example3), torch.tensor([[1., 0., 0., 1.]]))
    assert torch.equal(mlmc.thresholds.get(name="mcut").__call__(example4), torch.tensor([[1., 1., 1., 1., 0., 0.]]))
