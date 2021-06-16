from mlmc.representation import get
import torch
from tqdm import tqdm

def get_word_embedding_mean(words, model):
    """
    The word embeddings mean of all tokens in a list
    Args:
        words: List of words
        model: Model name (one of [glove50, glove100, glove200, glove300] or any of the models on https://huggingface.co/models

    Returns:
        Tensor of shape (1, embedding_dim)
    """
    emb, tok = get(model)
    transformed = tok(words)
    mask = (transformed!=0).int()
    embeddings = emb(transformed)
    return embeddings.sum(-2)/mask.sum(-1, keepdim=True)

def get_word_embeddings(words, model):
    """
    Get the sequence of word embeddings for a list of words. (This is basically a functional wrapper around
    the embedding class. Use only if the model to create word embeddings is used once and you need the memory freed)
    Args:
        words: List of words
        model: Model name (one of [glove50, glove100, glove200, glove300] or any of the models on https://huggingface.co/models

    Returns:
        Tensor of shape (1, sequence_length, embedding_dim)
    """
    emb, tok = get(model)
    transformed = tok(words)
    embeddings = emb(transformed)
    return embeddings
#
# def get_lm_repeated(classes, model, repeat=20):
#     emb, tok = get(model)
#     transformed = tok([" ".join(cls.split()*repeat) for cls in classes])
#     return emb(transformed)[1]
#
# def get_lm_generated(classes, model, num=10, device="cuda:0"):
#     with torch.no_grad():
#         from transformers import AutoModelWithLMHead, AutoTokenizer
#         gp2E = AutoModelWithLMHead.from_pretrained('gpt2').to(device)
#         gpt2T= AutoTokenizer.from_pretrained("gpt2")
#
#         generated_sentences = [[gpt2T.decode(x) for x in gp2E.generate(torch.tensor([gpt2T.encode((cls+" ")*3)]*num).to("cuda:0"), max_length=80)]
#          for cls in tqdm(classes, total=len(classes))
#          ]
#         emb, tok = get(model)
#         emb = emb.to(device)
#         embeddings = [torch.cat([emb(tok(s).to(device))[1] for s in cls]).mean(0) for cls in generated_sentences]
#     return torch.stack(embeddings,0)
#
# def get_graph_augmented(classes, model, graph, topk=20, batch_size=64, device="cuda:0"):
#     from ..graph.graph_loaders_elsevir import embed_align
#     doct = embed_align(classes, graph=graph, model="glove300", topk=topk, batch_size=batch_size, device=device)
#
#     emb, tok = get(model)
#     emb.to(device)
#     with torch.no_grad():
#         transformed = tok([cls + " | " + ". ".join(keywords) for cls, keywords in doct.items()])
#         embeddings = emb(transformed.to(device))[1]
#     return embeddings
