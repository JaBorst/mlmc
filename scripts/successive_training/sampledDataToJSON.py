import mlmc
import json

name="20newsgroup"
n=5

data = mlmc.data.get_multilabel_dataset(name)
samples = mlmc.data.successive_sampler(dataset=data["train"], classes = data["classes"], separate_dataset=n)

samples_dict = []

for dataset in samples[0]:
    samples_dict.append(
        {
            "train": dataset["train"].to_dict(),
            "test": dataset["test"].to_dict()
        }
    )

js = json.dumps(samples_dict, ensure_ascii=False)
with open('successive_datasets_'+name+'_'+str(n)+'.json', 'w', encoding='utf8') as json_file:
    json.dump(samples_dict, json_file, ensure_ascii=False)

