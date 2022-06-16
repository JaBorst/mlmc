import torch
import mlmc

def test_multilabeldataset_remove():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.remove(["a", "b"])

    assert d.x == ['3', '4'], "Removed a relevant data example"
    assert d.y == [["c"], ["d"]], "Removed wrong label sets"

    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.remove("b")
    assert d.x == ['1', '3', '4'], "Removed a relevant data example"
    assert d.y == [["a"], ["c"], ["d"]], "Removed wrong label sets"


def test_multilabeldataset_reduce():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.reduce(["a", "b"])

    assert d.x == ['1', '2'], "Removed a relevant data example"
    assert d.y == [["a", "b"], ["b"]], "Removed wrong label sets"

    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.reduce("b")
    assert d.x == ['1', '2'], "Removed a relevant data example"
    assert d.y == [["b"], ["b"]], "Removed wrong label sets"


def test_multilabeldataset_count():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    r = d.count(["a", "b"])
    assert r["a"] == 1 and r["b"] == 2, "Label count wrong"

    r = d.count(["e"])
    assert r["e"] == 0, "Counted a non occuring label wrong"


def test_multilabeldataset_density():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    r = d.density()
    assert (2 + 1 + 1 + 1) / 4 == r, "Calculated label density wrongly."


def test_multilabeldataset_map():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a", "b"], ["b"], ["c"], ["d"]],
                                                        classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.map({"b": "e"})
    assert d.x == ["1", "2", "3", "4"], "Falsely changed the data set when mapping label names."
    assert d.y == [["a", "e"], ["e"], ["c"], ["d"]], "False mapping of label names."


def test_multilabeldataset_add():
    d1 = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "4"],
                                                         y=[["a", "b"], ["b"], ["d"]],
                                                         classes={"a": 0, "b": 1, "d": 2})
    d2 = mlmc.data.dataset_classes.MultiLabelDataset(x=["2", "3", "4"],
                                                         y=[["e"], ["c"], ["d"]],
                                                         classes={"e": 0, "b": 1, "c": 2, "d": 3})

    d = d1 + d2
    d.one_hot = False
    assert set(d.classes.keys()) == {"a", "b", "c", "d", "e"}
    assert set(d.x) == {"1", "2", "3", "4"}
    assert set([tuple(sorted(x)) for x in d.y]) == set(
        [tuple(sorted(x)) for x in [["b", "a"], ["e", "b"], ["c"], ["d"]]])

def test_singlelabeldataset():
    try:
        mlmc.data.dataset_classes.SingleLabelDataset(x=["1", "2", "4"],
                                                         y=[["a", "b"], ["b"], ["d"]],
                                                         classes={"a": 0, "b": 1, "d": 2})
        assert False, "SingleLabelDataset  should not accept multiple labels per instance!"
    except AssertionError:
        pass

    d1 = mlmc.data.dataset_classes.SingleLabelDataset(x=["1", "2", "4"],
                                                          y=[["a"], ["b"], ["d"]],
                                                          classes={"a": 0, "b": 1, "d": 2})

    assert d1[0]["text"] == "1", "order of examples changed."
    assert d1[0]["labels"] == torch.tensor(0)
    assert d1[0]["labels"].shape == torch.Size([])

    for b in torch.utils.data.DataLoader(d1, 10): break
    assert (b["labels"] == torch.tensor([0, 1, 2])).all()


def test_multioutputsinglelabeldataset():
    try:
        mlmc.data.dataset_classes.MultiOutputSingleLabelDataset(x=["1", "2", "4"],
                                                                    y=[[["a"]], [["b"], ["c"]], [["d"], ["e"]]],
                                                                    classes={"a": 0, "b": 1, "d": 2})
        assert False, "SingleLabelDataset  should not accept multiple labels per instance!"
    except AssertionError:
        pass

    d1 = mlmc.data.dataset_classes.MultiOutputSingleLabelDataset(
        x=["1", "2", "4"],
        y=[[["a"], ["b"]], [["b"], ["c"]], [["d"], ["e"]]],
        classes={"a": 0, "b": 1, "c": 3, "d": 2, "e": 4})

    assert d1[0]["text"] == "1", "order of examples changed."
    assert (d1[0]["labels"] == torch.tensor([0, 1])).all()
    assert d1[0]["labels"].shape == torch.Size([len(d1.classes)])

    for b in torch.utils.data.DataLoader(d1, 10): break
    assert (b["labels"] == torch.tensor([[0, 1],
                                         [1, 3],
                                         [2, 4]])).all()

    d2 = mlmc.data.dataset_classes.MultiOutputSingleLabelDataset(
        x=["5", "6", "2"],
        y=[[["a"], ["b"]], [["b"], ["c"]], [["d"], ["e"]]],
        classes={"a": 0, "b": 1, "c": 3, "d": 2, "e": 4})

    assert len(d1 + d2) == 5


def test_multioutputmultilabeldataset():
    try:
        mlmc.data.dataset_classes.MultiOutputMultiLabelDataset(x=["1", "2", "4"],
                                                                   y=[[["a"]], [["b"], ["c"]], [["d"], ["e"]]],
                                                                   classes={"a": 0, "b": 1, "d": 2})
        assert False, "SingleLabelDataset  should not accept multiple labels per instance!"
    except AssertionError:
        pass

    d1 = mlmc.data.dataset_classes.MultiOutputMultiLabelDataset(
        x=["1", "2", "4"],
        y=[[["a", "b"], ["b"]], [["b", "d"], ["c"]], [["d", "e"], ["e"]]],
        classes={"a": 0, "b": 1, "c": 3, "d": 2, "e": 4})
    assert d1[0]["text"] == "1", "order of examples changed."
    assert (d1[0]["labels"][0] == torch.tensor([1, 1, 0, 0, 0])).all()
    assert (d1[0]["labels"][1] == torch.tensor([0, 1, 0, 0, 0])).all()
    assert d1[0]["labels"][0].shape == torch.Size([len(d1.classes[0])])
    assert d1[0]["labels"][1].shape == torch.Size([len(d1.classes[1])])

    for b in torch.utils.data.DataLoader(d1, 10): break
    assert (b["labels"][0] == torch.tensor([[1, 1, 0, 0, 0],
                                           [0, 1, 1, 0, 0],
                                           [0, 0, 1, 0, 1]])).all()

def test_augmentation():
    d = mlmc.data.dataset_classes.SingleLabelDataset(x=["the quick fox", "This is it", "A very short sentence"],
                                                           y=[["a"], ["b"], ["c"]],
                                                           classes={"a": 0, "b": 1, "d": 2})
    a = mlmc.data.Augmenter("sometimes", 0.1,0.1,0.1,0.1)
    d.generate(a, 0)
    assert len(d) == 3

    d.generate(a, 5)
    assert len(d) == 3 + 5*3
    assert len(d.x) == len(d.y)
