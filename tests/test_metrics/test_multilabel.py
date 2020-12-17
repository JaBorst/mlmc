from mlmc.metrics import multilabel
import torch

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
    d1, d2, d3 = multilabel.MultiLabelReport(), multilabel.MultiLabelReport(), multilabel.MultiLabelReport()
    d1.init(classes1, None)
    d2.init(classes1, None)
    d3.init(classes2, None)

    y_pred = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    y_truth = torch.tensor([[1, 1], [0, 1], [0, 1], [1, 0], [0, 1], [1, 1]])
    example1 = (classes1, y_truth, y_pred)
    y_pred = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 1], [1, 1]])
    y_truth = torch.tensor([[0, 1], [1, 0], [1, 1], [1, 1], [1, 1]])
    example2 = (classes1, y_truth, y_pred)
    y_pred = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_truth = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]])
    example3 = (classes2, y_truth, y_pred)

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

    d2.update(example2)
    assert d2.print() == results2

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

    d3.update(example3)
    assert d3.print() == results3
