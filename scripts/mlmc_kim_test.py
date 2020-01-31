
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







#
#
#
#
#
# # tc.predict([data["test"][0]["text"]] + [data["test"][1]["text"]])
# # _ = tc.fit(data["train"], mlmc.data.sample(data["test"],absolute=10000), epochs=100, batch_size=32)
# results1 = tc.evaluate(mlmc.data.sample(data["test"],absolute=1000), return_roc=True)
#
#
#
#
# results2 = tc.evaluate(mlmc.data.sample(data["test"],absolute=1000), return_roc=True)
# import matplotlib.pyplot as plt
# plt.plot(*results2["auc"][1])
# plt.plot(*results1["auc"][1]); plt.show()