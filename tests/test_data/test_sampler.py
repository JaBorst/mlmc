def test_MultiLabelDataset():
    d = mlmc.data.dataset_classes.MultiLabelDataset(x=["1", "2", "3", "4"],
                                                        y=[["a"],["b"],["c"],["d"]],
                                                        classes = {"a":0,"b":1,"c":2,"d":3})

    assert len(d) == 4
    assert list(d[0].keys()) == ["text", "labels"]
    assert tuple(d[0]["labels"].shape) == (4,)
    assert d[0]["labels"].tolist() == [1, 0, 0, 0]
    assert isinstance(d[0]["text"],str)


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
