
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import mlmc
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
tc = mlmc.models.KimCNN(classes=data["classes"],
                        mode="transformer",
                        transformer="bert",
                        #static= "/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.100d.txt",
                        optimizer=torch.optim.Adam,
                        optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
                        # optimizer_params={"lr": 0.000005, "betas": (0.9, 0.99)},
                        loss=torch.nn.BCEWithLogitsLoss,
                        device=device)
tc.fit(train=data["train"], valid=None, epochs=1, batch_size=32)













# tc.predict([data["test"][0]["text"]] + [data["test"][1]["text"]])
# _ = tc.fit(data["train"], mlmc.data.sample(data["test"],absolute=10000), epochs=100, batch_size=32)
results1 = tc.evaluate(mlmc.data.sample(data["test"],absolute=1000), return_roc=True)




results2 = tc.evaluate(mlmc.data.sample(data["test"],absolute=1000), return_roc=True)
import matplotlib.pyplot as plt
plt.plot(*results2["auc"][1])
plt.plot(*results1["auc"][1]); plt.show()