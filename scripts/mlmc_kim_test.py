
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import mlmc
data = mlmc.data.get_multilabel_dataset("blurbgenrecollection")



tc = mlmc.models.KimCNN(classes=data["classes"],
                        mode="transformer",
                        device=device)
history = tc.fit(train=data["train"], valid=data["valid"], epochs=1, batch_size=50)
evaluation = tc.evaluate(data["test"], batch_size=50)
prediction = tc.predict_dataset(mlmc.data.sampler(data["test"], absolute=100), batch_size=32, tr=0.65, method="mcut")





