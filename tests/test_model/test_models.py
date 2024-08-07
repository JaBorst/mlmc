import mlmc
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def assertion_function(model_type, **kwargs):

    classes = {"label_%i" % (i,): i for i in range(5)}
    data = mlmc.data.dataset_classes.MultiLabelDataset(
        x = ["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
        y = [["label_0", "label_4"], ["label_0", "label_2"], ["label_2", "label_3"], ["label_1", "label_4"], ["label_1", "label_4"]],
        classes = classes
    )


    model = model_type(classes=classes, **kwargs,
                               optimizer_params={"lr": 5})
    history = model.fit(train=data,  epochs=15, batch_size=32, patience=2, tolerance=0.1)

    assert len(history) < 15, "The probabilistic test of early stopping failed. Try to re-run. If the error persists, it's bad."

    loss, eval = model.evaluate(data)
    assert isinstance(eval, dict), "Return value of evaluate function failed."

    model = model_type(classes=classes, **kwargs, optimizer_params={"lr": 0.001})
    history = model.fit(train=data, epochs=5, batch_size=3)
    assert len(history["train"]["loss"]) == 5, "Number of Epochs not reached"

def test_KimCNN():
    assertion_function(model_type=mlmc.models.KimCNN, target="multi", filters=10, kernel_sizes=[3,4])

def test_XMLCNN():
    assertion_function(model_type=mlmc.models.XMLCNN, target="multi", filters=10, kernel_sizes=[3,4])
