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
    toks =  tok(words, padding=True, truncation=True,
                add_special_tokens=True, return_tensors='pt')
    transformed = toks["input_ids"]
    mask = toks["attention_mask"]
    embeddings = emb(transformed)[1]
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

def get_wikidata_desc(x, targets=3, sentences=2):
    """
    Queries wikidata using ID's to retrieve descriptions.

    :param x: List of ID's
    :return: Dictionary of input ID's corresponding to its wikidata descriptions
    """
    import wikipedia
    sites = [sum([wikipedia.search(k, results= targets) for k in l],[]) for l in x.values()]
    def _get_summaries(x):
        l = []
        for site in x:
            try: l.append(wikipedia.summary(site, sentences=sentences))
            except: pass
        return l
    descriptions = dict(zip(x.keys(), run_io_tasks_in_parallel(_get_summaries, sites)))
    return descriptions


from concurrent.futures import ThreadPoolExecutor

def run_io_tasks_in_parallel(func, args):
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(func, arg) for arg in args]
        l = []
        for running_task in running_tasks:
            l.append(running_task.result())
        return l
