import torch.cuda
import transformers
import tqdm
from transformers import pipeline
import random
import mlmc
import mlflow


dataset = "agnews"

def prompt(cls, example):
    if dataset == "dbpedia":
        cls = mlmc.data.label_dicts[dataset].get(cls,cls)
        s = "Wikipedia article:\nText: {example}\n\n"
        example = [example] if isinstance(example, str) else example
        s = "\n\n".join([s.replace("{example}", x) for x in example])
        return s + f"Wikipedia Article\n Category: {cls}\nText: "
    elif dataset =="agnews":
        cls = mlmc.data.label_dicts[dataset].get(cls,cls).replace("World", "World, General")
        s = "News article:\nText: {example}\n\n"
        example = [example] if isinstance(example, str) else example
        s = "\n\n".join([s.replace("{example}", x)[:512] for x in example])
        return s + f"news Article\n Category: {cls}\nText: "

def prompt_ctrl(cls, example):
    cls = mlmc.data.label_dicts[dataset].get(cls, cls)
    if dataset == "dbpedia":
        return f"Wikipedia {cls} {example[0].split(' ')[0]}"
    elif dataset =="agnews":
        return f"News {cls} {example[0].split(' ')[0]}"


def create_synth_dataset(model="microsoft/DialoGPT-large", classes=[], examples = [], k=32, n=4, s=8):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelWithLMHead.from_pretrained(model,pad_token_id=tokenizer.eos_token_id)
    text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1)
    x = []
    y = []
    for cls in classes.keys():
        for _ in tqdm.trange(int(k/n/s)):
            prefix_text =  prompt(cls, random.choices(examples, k=2))
            if n>1:
                generated_text = sum(text_generation([prefix_text]*n, max_length=512, do_sample=True, return_full_text=False, num_samples=n, num_return_sequences=s),[])
            else:
                generated_text = text_generation(prefix_text, max_length=512, do_sample=True, return_full_text=False, num_return_sequences=s)


            x.extend([x['generated_text'] for x in generated_text])
            y.extend([[cls]]*n*s)
    del text_generation
    del model
    torch.cuda.empty_cache()
    return mlmc.data.SingleLabelDataset(x=x, y=y, classes=classes)

gen_model="t5-large"


mlflow.set_tracking_uri("file:///home/jborst/generation")
id = mlflow.set_experiment("data")
with mlflow.start_run(run_name=f"{gen_model}_example_prompt"):

    mlflow.log_param("dataset", dataset)
    d = mlmc.data.get(dataset)
    u = mlmc.data.sampler(d["train"], absolute=32)
    usample = u.x[:32]
    lsample = u.y[:32]

    k=2#048
    s=32


    synth = create_synth_dataset(model=gen_model, classes = d["classes"], examples=usample, k=k, n=1, s=s)
    mlflow.log_param("K", k)
    mlflow.log_param("s", s)
    mlflow.log_param("gen_model", gen_model)

    import pickle
    with open("synth.data", "wb") as f: pickle.dump(synth, f)
    mlflow.log_artifact("synth.data")
