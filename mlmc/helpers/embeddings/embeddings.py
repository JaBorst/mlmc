import numpy as np
from tqdm import tqdm
from flair.data import Sentence
from flair.embeddings import BertEmbeddings,\
    ELMoEmbeddings,\
    WordEmbeddings, \
    FlairEmbeddings, \
    StackedEmbeddings, \
    BytePairEmbeddings, \
    RoBERTaEmbeddings


def embed(query, maxlen, model="glove"):
    if model=="bert": embedder = BertEmbeddings("bert-base-cased")
    if model=="bert1": embedder = BertEmbeddings("bert-base-cased", layers="-2")
    elif model=="distilbert": embedder = BertEmbeddings("distilbert-base-uncased")
    elif model=="roberta": embedder = RoBERTaEmbeddings()
    elif model=="elmo-medium": embedder = ELMoEmbeddings("medium")
    elif model== "elmo": embedder = ELMoEmbeddings("original")
    elif model=="glove": embedder = WordEmbeddings("glove")
    elif model=="glove_bytepair": embedder = StackedEmbeddings([WordEmbeddings("glove"), BytePairEmbeddings('en')])
    elif model=="glove_bytepair_de": embedder = StackedEmbeddings([WordEmbeddings("de"),BytePairEmbeddings("de")])
    elif model=="flair": embedder = StackedEmbeddings([FlairEmbeddings("news-forward"),FlairEmbeddings("news-backward")])
    elif model=="flair_glove": embedder = StackedEmbeddings([FlairEmbeddings("news-forward"),FlairEmbeddings("news-backward"),WordEmbeddings("glove")])
    elif model=="flair_glove_de": embedder = StackedEmbeddings([FlairEmbeddings("de-forward"),FlairEmbeddings("de-backward"),WordEmbeddings("de")])
    elif model=="bert_glove": embedder = StackedEmbeddings([BertEmbeddings("bert-base-cased"),WordEmbeddings("glove")])
    elif model=="bert_glove_de": embedder = StackedEmbeddings([BertEmbeddings("bert-base-german-cased"),WordEmbeddings("de")])
    else: Warning("Unknown Model")

    length = []
    query_embeddings = []
    for i, line in tqdm(enumerate(query), total=len(query)):
        line = line.replace("\n", "")
        sentence = Sentence(line)
        embedder.embed(sentence)
        embeddings = np.array([x.embedding.cpu().numpy() for x in sentence])
        query_embeddings.append(embeddings)
        length.append(len(sentence))

    result = np.full((len(query),maxlen, embedder.embedding_length), 0., dtype="float32")
    for i, e in enumerate(query_embeddings):
       result[i,:length[i],:] = e

    return result
