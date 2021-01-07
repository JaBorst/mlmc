from mlmc.metrics import ConfusionMatrix
import pytest
import torch

def test_confusionMatrixSingle():
    d = ConfusionMatrix(is_multilabel=False)
    y_truth1 = torch.tensor([[0], [1]])
    y_pred1 = torch.tensor([[0, 1], [1, 0]])
    example1 = (y_pred1, y_truth1)
    y_truth2 = torch.tensor([[0], [1], [2]])
    y_pred2 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    example2 = (y_pred2, y_truth2)
    y_pred3 = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    y_truth3 = torch.tensor([[0, 0], [0, 1], [1, 1], [0, 1], [0, 1], [1, 1]])
    example3 = (y_pred3, y_truth3)

    d.update(example1)
    assert torch.equal(torch.from_numpy(d.compute()), torch.tensor([[0, 1], [1, 0]]))

    d.reset()
    d.update(example2)
    assert torch.equal(torch.from_numpy(d.compute()), torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    d.reset()
    d.update(example3)
    with pytest.raises(ValueError):
        d.compute()

def test_confusionMatrixMulti():
    d = ConfusionMatrix(is_multilabel=True)
    y_pred1 = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    y_truth1 = torch.tensor([[0, 0], [0, 1], [1, 1], [0, 1], [0, 1], [1, 1]])
    example1 = (y_pred1, y_truth1)
    y_pred2 = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    y_truth2 = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    example2 = (y_pred2, y_truth2)
    y_truth3 = torch.tensor([[0], [1]])
    y_pred3 = torch.tensor([[0, 1], [1, 0]])
    example3 = (y_pred3, y_truth3)

    d.update(example1)
    assert torch.equal(torch.from_numpy(d.compute()), torch.tensor([[[2, 2], [1, 1]], [[0, 1], [1, 4]]]))

    d.reset()
    d.update(example2)
    assert torch.equal(torch.from_numpy(d.compute()), torch.tensor([[[4, 0], [0, 0]], [[4, 0], [0, 0]], [[4, 0], [0, 0]]]))

    d.reset()
    d.update(example3)
    with pytest.raises(ValueError):
        d.compute()
