import torch
import mlmc

def test_multilabeldataset_remove():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    d.remove(["a", "b"])

    assert d.x == ['3', '4'], "Removed a relevant data example"
    assert d.y == [["c"], ["d"]], "Removed wrong label sets"

    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    d.remove("b")
    assert d.x == ['1', '3', '4'], "Removed a relevant data example"
    assert d.y == [["a"], ["c"], ["d"]], "Removed wrong label sets"


def test_multilabeldataset_reduce():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    d.reduce(["a", "b"])

    assert d.x == ['1', '2'], "Removed a relevant data example"
    assert d.y == [["a", "b"], ["b"]], "Removed wrong label sets"

    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    d.reduce("b")
    assert d.x == ['1', '2'], "Removed a relevant data example"
    assert d.y == [["b"], ["b"]], "Removed wrong label sets"


def test_multilabeldataset_count():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    r = d.count(["a", "b"])
    assert r["a"] == 1 and r["b"] == 2, "Label count wrong"

    r = d.count(["e"])
    assert r["e"] == 0, "Counted a non occuring label wrong"


def test_multilabeldataset_density():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    r = d.density()
    assert (2 + 1 + 1 + 1) / 4 == r, "Calculated label density wrongly."


def test_multilabeldataset_map():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                     hypothesis = ["1", "2", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes=["a", "b", "c", "d"])

    d.map({"b": "e"})
    assert d.x == ["1", "2", "3", "4"], "Falsely changed the data set when mapping label names."
    assert d.y == [["a", "e"], ["e"], ["c"], ["d"]], "False mapping of label names."


def test_multilabeldataset_add():
    d1 = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "4"],
                                                         y=[["a", "b"], ["b"], ["d"]],
                                                         classes=["a", "b", "d"])
    d2 = mlmc.data.dataset_classes.MultiLabelDataset(x=["2", "3", "4"],
                                                         y=[["e"], ["c"], ["d"]],
                                                         classes=["e", "b", "c", "d"])

    d = d1 + d2
    d.one_hot = False
    assert set(d.classes.keys()) == {"a", "b", "c", "d", "e"}
    assert set(d.x) == {"1", "2", "3", "4"}
    assert set([tuple(sorted(x)) for x in d.y]) == set(
        [tuple(sorted(x)) for x in [["b", "a"], ["e", "b"], ["c"], ["d"]]])

    # With Hypothesis
    d1 = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "4"],
                                                     hypothesis=["1", "2", "4"],
                                                     y=[["a", "b"], ["b"], ["d"]],
                                                     classes=["a", "b", "d"])
    d2 = mlmc.data.dataset_classes.MultiLabelDataset(x=["2", "3", "4"],
                                                     hypothesis=["1", "2", "4"],
                                                     y=[["e"], ["c"], ["d"]],
                                                     classes=["e", "b", "c", "d"])

    d = d1 + d2
    d.one_hot = False
    assert set(d.classes.keys()) == {"a", "b", "c", "d", "e"}
    assert set(d.x) == {"1", "2", "3", "4"}
    assert set([tuple(sorted(x)) for x in d.y]) == set(
        [tuple(sorted(x)) for x in [["b", "a"], ["b"], ["c"], ["d"], ["e"]]])

def test_singlelabeldataset():
    try:
        mlmc.data.dataset_classes.SingleLabelDataset(x=["1", "2", "4"],
                                                         y=[["a", "b"], ["b"], ["d"]],
                                                         classes=["a", "b", "d"])
        assert False, "SingleLabelDataset  should not accept multiple labels per instance!"
    except AssertionError:
        pass

    d1 = mlmc.data.dataset_classes.SingleLabelDataset(x=["1", "2", "4"],
                                                          y=[["a"], ["b"], ["d"]],
                                                          classes=["a", "b", "d"])

    assert d1[0]["text"] == "1", "order of examples changed."
    assert d1[0]["labels"] == torch.tensor(0)
    assert d1[0]["labels"].shape == torch.Size([])

    for b in torch.utils.data.DataLoader(d1, 10): break
    assert (b["labels"] == torch.tensor([0, 1, 2])).all()


def test_augmentation():
    d = mlmc.data.dataset_classes.SingleLabelDataset(x=["the quick fox", "This is it", "A very short sentence"],
                                                           y=[["a"], ["b"], ["c"]],
                                                           classes=["a", "b", "d"])
    a = mlmc.data.Augmenter("sometimes", 0.1,0.1,0.1,0.1)
    d.generate(a, 0)
    assert len(d) == 3

    d.generate(a, 5)
    assert len(d) == 3 + 5*3
    assert len(d.x) == len(d.y)
