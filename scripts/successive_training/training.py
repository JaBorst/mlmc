import mlmc
import torch

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
name = "20newsgroup" #  "blurbgenrecollection"
epochs = 10
batch_size=32
n=5

data = mlmc.data.get_multilabel_dataset(name)
samples, ind = mlmc.data.successive_sampler(dataset=data["train"], classes = data["classes"], separate_dataset=n)

for i in range(len(samples)-1):
    model = mlmc.models.KimCNN(samples[i]["train"].classes,
                               representation ="roberta",
                               device=device)
    # Set the precision of the metrics (round(..., model.PRECISION_DIGITS))
    model.PRECISION_DIGITS=4
    # Fit the current subset
    history = model.fit(train=samples[i]["train"],
                        valid=samples[i+1]["test"],
                        epochs=epochs,
                        batch_size=batch_size,
                        valid_batch_size=batch_size)

    # Evaluate Test set after Trainin and get the per class F1 (return_report=True) and the values to plot and AUC_ROC
    # (return_roc=True)
    evaluation = model.evaluate(samples[i+1]["test"], batch_size=batch_size, return_roc=True, return_report=True)


    # Prediction.
    #           If method="hard":  Threshold 0.65.  every label with confidence >0.65 is set as prediction.
    #           if method="mcut":  The greatest distance in confidence of two next-to-each other ranked labels is used
    #                              to cut predicted from non-predicted labels
    prediction = model.predict_dataset(samples[i+1]["test"],batch_size=32, tr=0.65, method="hard")
