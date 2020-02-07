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