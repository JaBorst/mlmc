import mlmc.data as md


def test_MultiLabelDataset_remove():
    d = md.MultiLabelDataset(x=["1","2","3","4"],
                             y=[["a", "b"],["b"],["c"],["d"]],
                             classes = {"a":0,"b":1,"c":2,"d":3})

    d.remove(["a", "b"])

    assert d.x == ['3', '4'], "Removed a relevant data example"
    assert d.y == [["c"], ["d"]], "Removed wrong label sets"

    d = md.MultiLabelDataset(x=["1", "2", "3", "4"],
                             y=[["a", "b"], ["b"], ["c"], ["d"]],
                             classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.remove("b")
    assert d.x == ['1', '3', '4'], "Removed a relevant data example"
    assert d.y == [["a"], ["c"], ["d"]], "Removed wrong label sets"


def test_MultiLabelDataset_reduce():
    d = md.MultiLabelDataset(x=["1","2","3","4"],
                             y=[["a", "b"],["b"],["c"],["d"]],
                             classes = {"a":0,"b":1,"c":2,"d":3})

    d.reduce(["a", "b"])

    assert d.x == ['1', '2'], "Removed a relevant data example"
    assert d.y == [["a","b"], ["b"]], "Removed wrong label sets"

    d = md.MultiLabelDataset(x=["1", "2", "3", "4"],
                             y=[["a", "b"], ["b"], ["c"], ["d"]],
                             classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.reduce("b")
    assert d.x == ['1', '2'], "Removed a relevant data example"
    assert d.y == [["b"], ["b"]], "Removed wrong label sets"

def test_MultiLabelDataset_count():
    d = md.MultiLabelDataset(x=["1","2","3","4"],
                             y=[["a", "b"],["b"],["c"],["d"]],
                             classes = {"a":0,"b":1,"c":2,"d":3})

    r = d.count(["a", "b"])
    assert r["a"] == 1 and r["b"] == 2, "Label count wrong"

    r = d.count(["e"])
    assert r["e"] == 0, "Counted a non occuring label wrong"


def test_MultiLabelDataset_density():
    d = md.MultiLabelDataset(x=["1","2","3","4"],
                             y=[["a", "b"],["b"],["c"],["d"]],
                             classes = {"a":0,"b":1,"c":2,"d":3})

    r = d.density()
    assert (2+1+1+1)/4 == r, "Calculated label density wrongly."


def test_MultiLabelDataset_map():
    d = md.MultiLabelDataset(x=["1", "2", "3", "4"],
                             y=[["a", "b"], ["b"], ["c"], ["d"]],
                             classes={"a": 0, "b": 1, "c": 2, "d": 3})

    d.map({"b": "e"})
    assert d.x==["1", "2", "3", "4"], "Falsely changed the data set when mapping label names."
    assert d.y == [["a", "e"], ["e"], ["c"], ["d"]] , "False mapping of label names."


def test_MultiLabelDataset_add():
    d1 = md.MultiLabelDataset(x=["1", "2", "4"],
                             y=[["a", "b"], ["b"], ["d"]],
                             classes={"a": 0, "b": 1, "d":2})
    d2 = md.MultiLabelDataset(x=["2", "3", "4"],
                             y=[["e"], ["c"], ["d"]],
                             classes={"e": 0, "b": 1, "c": 2, "d": 3})

    d = d2 + d1
    d.one_hot=False
    assert set(d.classes.keys()) == {"a", "b", "c", "d", "e"}
    assert set(d.x) == {"1", "2", "3", "4"}
    assert set([ tuple(sorted(x)) for x in d.y]) == set([tuple(sorted(x)) for x in [["b","a"], ["e", "b"], ["c"], ["d"]]])

