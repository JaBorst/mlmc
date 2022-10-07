import mlmc

classes = {"label_0": 0, "label_1": 1, "label_2": 2, "label_3": 3, "label_4": 4}

ml = mlmc.data.dataset_classes.MultiLabelDataset(
    x=["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
    y=[["label_0", "label_4"], ["label_0", "label_2"], ["label_2", "label_3"], ["label_1", "label_4"],
       ["label_1", "label_4"]],
    classes=classes
)
sl = mlmc.data.dataset_classes.SingleLabelDataset(
    x=["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
    y=[["label_0"], ["label_0"], ["label_2"], ["label_1"], ["label_1"]],
    classes=classes
)

en = mlmc.data.dataset_classes.EntailmentDataset(
    x=["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
    hypothesis = ["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
    y=[["contradiction"], ["contradiction"], ["contradiction"], ["neutral"], ["contradiction"]],
    classes={'contradiction': 0, 'neutral': 1, 'contradiction': 2}
)
abc = mlmc.data.dataset_classes.ABCDataset(
    x=["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
    hypothesis = ["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
    y=[["label_0"], ["label_0"], ["label_2"], ["label_1"], ["label_1"]],
    classes=classes
)


def f(model):
    m = model(classes = classes, target="entailment", finetune="all", device="cuda:0")
    try:
        m.single()
        m.evaluate(sl)
        m.fit(sl)
    except:
        assert False, f"Error in {model} for single label"
    try:
        m.multi()
        m.evaluate(ml)
        m.fit(ml)
    except:
        assert False, f"Error in {model} for multi label"
    try:
        m.abc(sformatter = lambda asp, cls: f"the aspect {asp} is {cls}")
        m.evaluate(abc)
        m.fit(abc)
    except:
        assert False, f"Error in {model} for single label"
    try:
        m.entailment(sformatter = lambda  cls: f"this is {cls}")
        m.create_labels({'contradiction': 0, 'neutral': 1, 'entailment': 2})
        m.evaluate(en)
        m.fit(en)
    except:
        assert False, f"Error in {model} for single label"


def test_encoder(): f(mlmc.models.Encoder)
def test_siamese(): f(mlmc.models.Siamese)

test_encoder()