from mlmc.metrics import multilabel
import torch
import pytest

def test_AUC():
    d = multilabel.AUC_ROC()
    d.__init__(return_roc = False)

    y_score = torch.tensor([[0.1], [0.4], [0.35], [0.8]])
    y_truth = torch.tensor([[0], [0], [1], [1]])
    y_pred = torch.tensor([[0], [0], [1], [1]])
    example1 = (y_score, y_truth, y_pred)
    y_score = torch.tensor([[0.1, 0.3, 0.8, 0.8], [0.2, 0.1, 0.2, 0.9]])
    y_truth = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    y_pred = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    example2 = (y_score, y_truth, y_pred)

    d.update(example1)
    assert d.compute() == 0.75

    d.reset()
    d.update(example2)
    assert d.compute() == 0.5

def test_MultiLabelReport():
    classes1 = {"label_%i" % (i,): i for i in range(2)}
    classes2 = {"label_%i" % (i,): i for i in range(3)}
    d1, d2 = multilabel.MultiLabelReport(), multilabel.MultiLabelReport(is_multilabel = False)

    #Multi-label examples
    y_pred1 = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    y_truth1 = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    example1 = (classes1, y_truth1, y_pred1)
    y_pred2 = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 1], [1, 1]])
    y_truth2 = torch.tensor([[0, 1], [1, 0], [1, 1], [1, 1], [1, 1]])
    example2 = (classes1, y_truth2, y_pred2)
    y_pred3 = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_truth3 = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    example3 = (classes2, y_truth3, y_pred3)

    #Single-label examples
    y_truth4 = torch.tensor([1, 0, 2])
    y_pred4 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    example4 = (classes2, y_truth4, y_pred4)
    y_truth5 = torch.tensor([0, 1, 1, 0])
    y_pred5 = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    example5 = (classes1, y_truth5, y_pred5)

    results1 = {'micro avg': {
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1-score': 1.0,
                    'support': 8},
                'macro avg': {
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1-score': 1.0,
                    'support': 8}}

    d1.init(classes1, None)
    d1.update(example1)
    assert d1.print() == results1

    results2 = {'micro avg': {
                    'precision': 0.75,
                    'recall': 0.75,
                    'f1-score': 0.75,
                    'support': 8},
                'macro avg': {
                    'precision': 0.75,
                    'recall': 0.75,
                    'f1-score': 0.75,
                    'support': 8}}

    d1.reset()
    d1.init(classes1, None)
    d1.update(example2)
    assert d1.print() == results2

    results3 = {'micro avg': {
                    'precision': 0.5,
                    'recall': 0.5,
                    'f1-score': 0.5,
                    'support': 4},
                'macro avg': {
                    'precision': 0.5,
                    'recall': 0.5,
                    'f1-score': 0.5,
                    'support': 4}}

    d1.reset()
    d1.init(classes2, None)
    d1.update(example3)
    assert d1.print() == results3

    d1.reset()
    d1.update(example4)
    with pytest.raises(ValueError):
        d1.print()

    results4 = {'micro avg': {
                    'precision': 0.3333333333333333,
                    'recall': 0.3333333333333333,
                    'f1-score': 0.3333333333333333,
                    'support': 3},
                'macro avg': {
                    'precision': 0.3333333333333333,
                    'recall': 0.3333333333333333,
                    'f1-score': 0.3333333333333333,
                    'support': 3}}

    d2.init(classes2, None)
    d2.update(example4)
    assert d2.print() == results4

    d2.reset()
    d2.init(classes1, None)
    d2.update(example5)
    assert d2.print() == results3

    d2.reset()
    d2.update(example1)
    with pytest.raises(ValueError):
        d2.print()
