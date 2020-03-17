import mlmc.data as md


def test_MultiLabelDataset():
    d = md.MultiLabelDataset(x=[1,2,3,4],
                             y=[["a"],["b"],["c"],["d"]],
                             classes = {"a":0,"b":1,"c":2,"d":3})

    assert len(d) == 4
    assert list(d[0].keys()) == ["text", "labels"]
    assert tuple(d[0]["labels"].shape) == (4,)
    assert d[0]["labels"].tolist() == [1, 0, 0, 0]
    assert len(d[0]["text"]) == 1


def test_MultiLabelDataset_add():
    a = md.MultiLabelDataset(x=[1,2,3,4],
                             y=[["a"], ["b"], ["c"], ["d"]],
                             classes = {"a":0,"b":1,"c":2,"d":3})

    b = md.MultiLabelDataset(x=[1,4,5],
                             y=[["a", "e"], ["d"], ["e"]],
                             classes = {"a":0,"d":1,"e":2})

    c = a + b

    assert set(c.classes.keys()) == set(list(a.classes.keys()) + list(b.classes.keys()))
    assert c.x == [1,2,3,4,5]
    assert c.y == [["a", "e"], ["b"], ["c"], ["d"], ["e"]]
    assert set(list(c.classes.keys())) == set(["a", "b", "c", "d", "e"])
