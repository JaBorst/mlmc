import mlmc

def test_random_sampler():
    source = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4", "5"],
                                                        y=[["a"],["b"],["c"],["d"],["d"]],
                                                        classes = {"a":0,"b":1,"c":2,"d":3})
    d = mlmc.data.sampler(source, absolute=3)
    assert len(d) == 3, f"sampling with absolute=3 produces dataset of size {len(d)}"
    d = mlmc.data.sampler(source, fraction=0.2)
    assert len(d) == 1, f"sampling with absolute=3 produces dataset of size {len(d)}"

    source = mlmc.data.dataset_classes.SingleLabelDataset(x=["1", "2", "3", "4", "5"],
                                                         y=[["a"], ["b"], ["c"], ["d"], ["d"]],
                                                         classes={"a": 0, "b": 1, "c": 2, "d": 3})
    d = mlmc.data.sampler(source, absolute=3)
    assert len(d) == 3, f"sampling with absolute=3 produces dataset of size {len(d)}"
    d = mlmc.data.sampler(source, fraction=0.2)
    assert len(d) == 1, f"sampling with absolute=3 produces dataset of size {len(d)}"


    source = mlmc.data.dataset_classes.ABCDataset(x=["1", "2", "3", "4", "5"],
                                                         hypothesis=["1", "2", "3", "4", "5"],
                                                         y=[["a"], ["b"], ["c"], ["d"], ["d"]],
                                                         classes={"a": 0, "b": 1, "c": 2, "d": 3})
    d = mlmc.data.sampler(source, absolute=3)
    assert len(d) == 3, f"sampling with absolute=3 produces dataset of size {len(d)}"
    d = mlmc.data.sampler(source, fraction=0.2)
    assert len(d) == 1, f"sampling with absolute=3 produces dataset of size {len(d)}"

    source = mlmc.data.dataset_classes.EntailmentDataset(x=["1", "2", "3", "4", "5"],
                                                         hypothesis=["1", "2", "3", "4", "5"],
                                                         y=[["a"], ["b"], ["c"], ["d"], ["d"]],
                                                         classes={"a": 0, "b": 1, "c": 2, "d": 3})
    d = mlmc.data.sampler(source, absolute=3)
    assert len(d) == 3, f"sampling with absolute=3 produces dataset of size {len(d)}"
    d = mlmc.data.sampler(source, fraction=0.2)
    assert len(d) == 1, f"sampling with absolute=3 produces dataset of size {len(d)}"



def test_MultiLabelDataset_add():
    a = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a"], ["b"], ["c"], ["d"]],
                                                        classes = {"a":0,"b":1,"c":2,"d":3})

    b = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "4", "5"],
                                                        y=[["a", "e"], ["d"], ["e"]],
                                                        classes = {"a":0,"d":1,"e":2})

    c = a + b

    assert set(c.classes.keys()) == set(list(a.classes.keys()) + list(b.classes.keys()))
    assert len(set(c.x)) == len(c.x), "Duplicates in added MultilabelDatasets"
    assert set(c.x) == set(["1","2","3","4","5"])
    # assert all([x in c.y for x in  [["a", "e"], ["b"], ["c"], ["d"], ["e"]]])
    assert set(list(c.classes.keys())) == set(["a", "b", "c", "d", "e"])
