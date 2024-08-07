import mlmc
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def assertion_function(model_type, **kwargs):
    classes = {"label_%i" % (i,): i for i in range(5)}
    data = mlmc.data.dataset_classes.SingleLabelDataset(
        x = ["Text 1 example", "text 2 example 2", "text 2 example 2", "text 2 example 2", "text 2 example 2"],
        y = [["label_0"], [ "label_2"], [ "label_3"], ["label_1"], ["label_4"]],
        classes = classes
    )
    model = model_type(classes=classes, **kwargs, target="single", optimizer_params={"lr": 0.001})
    history = model.fit(train=data, epochs=5, batch_size=3)
    assert len(history["train"]["loss"]) == 5, "Number of Epochs not reached"

def test_KimCNN():
    assertion_function(model_type=mlmc.models.KimCNN,  filters=10, kernel_sizes=[3,4])

def test_XMLCNN():
    assertion_function(model_type=mlmc.models.XMLCNN,  filters=10, kernel_sizes=[3,4])

def test_LSAN_transformer():
    assertion_function(model_type=mlmc.models.LSAN, label_model="google/bert_uncased_L-2_H-128_A-2", hidden_representations=10, d_a=10)
