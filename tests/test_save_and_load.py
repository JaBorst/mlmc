import tempfile
from pathlib import Path
import mlmc
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compare_models(model_1, model_2):
    for layer_1, layer_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(layer_1[1], layer_2[1]):
            pass
        else:
            return False
    return True

def save_and_load(model):
    with tempfile.TemporaryDirectory() as tmp:
        model_path = tmp + "/test.pt"
        print("Saving model to " + model_path)
        mlmc.save(model, model_path)
        assert(Path(model_path).exists()), "Model didn't save"
        print("Loading model from " + model_path)
        loaded_model = mlmc.load(model_path)
        assert(loaded_model is not None), "Saved model didn't load"
        assert(compare_models(model, loaded_model) == True), "Models don't match"
        return loaded_model

def assertion_function(model_type, **kwargs):

    data = mlmc.data.dataset_classes.MultiLabelDataset(
        x = ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
        y = [["label_0", "label_4"], ["label_0", "label_2"], ["label_2", "label_3"], ["label_1", "label_4"], ["label_1", "label_4"]],
        classes = {"label_%i" % (i,): i for i in range(5)}
    )



    model = model_type(classes=data.classes, **kwargs)

    #test saving and loading before training
    loaded_model = save_and_load(model)

    model.fit(train=data, epochs=5, batch_size=3)
    assert(compare_models(model, loaded_model) == False), "No parameter change after training"

    #test saving and loading after training
    save_and_load(model)

def test_KimCNN():
    assertion_function(model_type=mlmc.models.KimCNN, target="multi", representation="google/bert_uncased_L-2_H-128_A-2", kernel_sizes=(2,3))

def test_LSAN():
    assertion_function(model_type=mlmc.models.LSAN, target="multi",representation="google/bert_uncased_L-2_H-128_A-2", hidden_representations=10, d_a=10)

def test_Transformer():
    assertion_function(model_type=mlmc.models.Transformer, target="multi", representation="google/bert_uncased_L-2_H-128_A-2")

def test_XMLCNN():
    assertion_function(model_type=mlmc.models.XMLCNN, target="multi", representation="google/bert_uncased_L-2_H-128_A-2", kernel_sizes=(2,3))
