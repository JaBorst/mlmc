from mlmc.representation import get
import torch
from tqdm import tqdm

def get_word_embedding_mean(classes, model):
    emb, tok = get(model)
    transformed = tok(classes)
    mask = (transformed!=0).int()
    embeddings = emb(transformed)
    return embeddings.sum(-2)/mask.sum(-1, keepdim=True)

def get_lm_repeated(classes, model, repeat=20):
    emb, tok = get(model)
    transformed = tok([" ".join(cls.split()*repeat) for cls in classes])
    return emb(transformed)[1]

def get_lm_generated(classes, model, num=10, device="cuda:0"):
    with torch.no_grad():
        from transformers import AutoModelWithLMHead, AutoTokenizer
        gp2E = AutoModelWithLMHead.from_pretrained('gpt2').to(device)
        gpt2T= AutoTokenizer.from_pretrained("gpt2")

        generated_sentences = [[gpt2T.decode(x) for x in gp2E.generate(torch.tensor([gpt2T.encode((cls+" ")*3)]*num).to("cuda:0"), max_length=80)]
         for cls in tqdm(classes, total=len(classes))
         ]
        emb, tok = get(model)
        emb = emb.to(device)
        embeddings = [torch.cat([emb(tok(s).to(device))[1] for s in cls]).mean(0) for cls in generated_sentences]
    return torch.stack(embeddings,0)

