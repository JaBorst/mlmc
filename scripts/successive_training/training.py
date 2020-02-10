import mlmc
import torch
import pathlib
import json
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
name = "20newsgroup"
epochs = 15
batch_size=50
n=5
path = pathlib.Path(name+"_first_run")



if not path.exists():
    path.mkdir()


data = mlmc.data.get_multilabel_dataset(name)
samples, ind = mlmc.data.successive_sampler(dataset=data["train"], classes = data["classes"], separate_dataset=n, reindex_classes=False)

# Save the current dataset versions
samples_dict = []

for dataset in samples:
    samples_dict.append(
        {
            "train": dataset["train"].to_dict(),
            "test": dataset["test"].to_dict()
        }
    )

js = json.dumps(samples_dict, ensure_ascii=False)
with open(path/('successive_datasets_'+name+'_'+str(n)+'.json'), 'w', encoding='utf8') as json_file:
    json.dump(samples_dict, json_file, ensure_ascii=False)



sample_evaluations = []
testset_evaluations = []

for i in range(len(samples)):
    print("##########################################################\n"
          " Sample "+str(i)+"\n"
          "##########################################################\n")
    # test sample of the following sample for the calculation of acceptance rates. For the last in the list, it will be the test sample of the same sample
    test_i = min(len(samples)-1,i+1)

    #Create a new model with standard Arguments
    model = mlmc.models.KimCNN(data["classes"],
                               mode="transformer",
                               representation="roberta",
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
    mlmc.save(model, path / ("model_" + str(i) + ".pt"), only_inference=False)

    # Evaluate Test set after Training and get the per class F1 (return_report=True) and the values to plot and AUC_ROC
    # (return_roc=True) for overall quality
    testset_evaluations.append(
        model.evaluate_classes(
            classes_subset=samples[i]["train"].occuring_classes,
            data=data["test"],
            batch_size=batch_size,
            return_roc=True,
            return_report=True
        )
    )

    pr_perclass = {k:v for k,v in testset_evaluations[-1]["report"].items() if k in samples[i]["train"].occuring_classes}
    testset_evaluations[-1]["report"] = pr_perclass

    # Prediction on the testset
    #           If method="hard":  Threshold 0.5.  every label with confidence >0.5 is set as prediction. (Can in some cases lead to empty predictions)
    #           if method="mcut":  The greatest distance in confidence of two next-to-each other ranked labels is used
    #                              to cut predicted from non-predicted labels
    testset_predictions = model.predict_dataset(data["test"], batch_size=2*batch_size, tr=0.5, method="mcut")
    with open(path / ("testset_predictions_" + str(i) + ".txt"), "w") as f:
        for l in testset_predictions: f.write(",".join(l) + "\n")


    #--------------------------------
    #   Next Sample Evaluations
    #--------------------------------

    sample_evaluations.append(
        model.evaluate_classes(
            classes_subset=samples[i]["train"].occuring_classes,
            data=samples[test_i]["train"],
            batch_size=batch_size,
            return_roc=True,
            return_report=True
        )
    )

    # Prediction on the i+1 dataset "test" to calculate for acceptance rate ("active learning")
    #           If method="hard":  Threshold 0.5. every label with confidence >0.5 is set as prediction. (Can in some cases lead to empty predictions)
    #           if method="mcut":  The greatest distance in confidence of two next-to-each other ranked labels is used
    #                              to cut predicted from non-predicted labels
    predictions = model.predict_dataset(samples[test_i]["test"], batch_size=2 * batch_size, tr=0.5, method="mcut")
    with open(path / ("prediction_" + str(i) + ".txt"), "w") as f:
        for l in predictions: f.write(",".join(l) + "\n")


    # Check Acceptance rate for the additional data points in sample i + 1
    accepted = [any([x in t for x in p]) for p, t in zip(predictions,samples[test_i]["test"].y)]
    acceptance_rate = sum(accepted) / len(accepted)

    sample_evaluations[-1]["acceptance_rate"]=acceptance_rate

    exact_matches = [all([x in [y for y in t if y in samples[test_i]["train"].occuring_classes] for x in p]) for p, t in zip(predictions,samples[test_i]["test"].y)]
    exact_matches = sum(exact_matches) / len(exact_matches)
    sample_evaluations[-1]["exact_matches"] = exact_matches


testset_reports = [d["report"] for d in testset_evaluations]
for d in testset_evaluations: del d["report"]
with open(path/("testset_reports.json"), "w")as f: f.write(json.dumps(testset_reports))

for d in testset_evaluations: d["auc_roc"]=d["auc"][0]
testset_roc = [{"x": d["auc"][1][0], "y": d["auc"][1][1]} for d in testset_evaluations]
for d in testset_evaluations: del d["auc"]
with open(path/"testset_roc.json","w") as f: f.write(json.dumps(testset_roc))

sample_reports = [d["report"] for d in sample_evaluations]
for d in sample_evaluations: del d["report"]
with open(path/("sample_reports.json"), "w")as f: f.write(json.dumps(sample_reports))

for d in sample_evaluations: d["auc_roc"]=d["auc"][0]
sample_roc = [{"x": d["auc"][1][0], "y": d["auc"][1][1]} for d in sample_evaluations]
for d in sample_evaluations: del d["auc"]
with open(path/"sample_roc.json","w") as f: f.write(json.dumps(sample_roc))

testset_df = pd.DataFrame(testset_evaluations)
testset_df.to_csv(path/("testset_evaluations.csv"))

sample_df = pd.DataFrame(sample_evaluations)
sample_df.to_csv(path/("sample_evaluations.csv"))

