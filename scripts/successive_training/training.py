import mlmc
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
name = "movies_summaries" #  "blurbgenrecollection"
epochs = 15
batch_size=50
n=3

data = mlmc.data.get_multilabel_dataset(name)
samples, ind = mlmc.data.successive_sampler(dataset=data["train"], classes = data["classes"], separate_dataset=n, reindex_classes=False)

for i in range(len(samples)-1):
    #Create a new model with standard Arguments
    model = mlmc.models.KimCNN(data["classes"],
                               mode="transformer",
                               representation = "roberta",
                               device=device)
    # Set the precision of the metrics (round(..., model.PRECISION_DIGITS))
    model.PRECISION_DIGITS=8
    # Fit the current subset
    history = model.fit(train=data["train"],
                        epochs=epochs,
                        batch_size=batch_size,
                        valid_batch_size=batch_size,
                        classes_subset=samples[i]["train"].occuring_classes
                        )

    # Evaluate Test set after Training and get the per class F1 (return_report=True) and the values to plot and AUC_ROC
    # (return_roc=True) for overall quality
    evaluation = model.evaluate_classes(
        classes_subset=samples[i]["train"].occuring_classes,
        data=data["test"],
        batch_size=batch_size,
        return_roc=True,
        return_report=True)


    # Prediction on the i+1 dataset to calculate for acceptance rate ("active learning")
    #           If method="hard":  Threshold 0.65.  every label with confidence >0.65 is set as prediction.
    #           if method="mcut":  The greatest distance in confidence of two next-to-each other ranked labels is used
    #                              to cut predicted from non-predicted labels
    prediction = model.predict_dataset(samples[i+1]["test"],batch_size=32, tr=0.65, method="mcut")

    # Check Acceptance rate for the additional classes in sample i + 1
    accepted = [any([x in t for x in p]) for p, t in zip(prediction,samples[i+1]["test"].y)]
    acceptance_rate = sum(accepted) / len(accepted)

    exact_matches = [all([x in [y for y in t if y in samples[i+1]["train"].occuring_classes] for x in p]) for p, t in zip(prediction,samples[i+1]["test"].y)]
    exact_matches = sum(exact_matches) / len(exact_matches)