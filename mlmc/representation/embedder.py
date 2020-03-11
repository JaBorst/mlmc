from typing import List

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..representation import get


class _temporal_dataset(Dataset):
    def __init__(self, l):
        self.l = l
    def __len__(self):
        return len(self.l)
    def __getitem__(self, item):
        return self.l[item]

class Embedder:
    """
    A class for embedding lists of sentences.
    """
    def __init__(self, representation, device="cpu", return_device="cpu"):
        """
        Holding an embedding to embed lists of sentences with huggingface or glove.

        This class returns an embedding vector for every whitespace separated token in the data.
        (Tested only on all BERT-like models which use a prefixe for whitespaces.)
        For now only the first_token policy is supported.

        ToDo:
            - Support Pooling method

        :param representation: Name of the representation (language model or glove vectors)
        :param device: The device to use for the computation of the embeddings
        :param return_device: The device to store the results on ( For large data it is adviced to leave this on cpu, for smaller amount of data that will be used for further computation
        it can be useful to leave it on the device for further processing
        """
        self.representation = representation
        self.emb, self.tok = get(self.representation)
        self.device = device
        self.emb = self.emb.to(self.device)
        self.return_device = return_device

    def embed(self, sentences: List, pad = None):
        """
        Embedding method for a list of sentences.
        :param sentences:  List of sentences
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: if pad is None a list of embeddings (with varying lengths) is returned, is pad is set a tensor of (num_sentences, pad, embedding_size) will be returned.
        """
        t, ind = self.tok(sentences, return_start=True)
        import torch
        with torch.no_grad():
            embeddings = self.emb(t.to(self.device))[0].to(self.return_device)
        embeddings = [e[i]for e, i in zip(embeddings, ind)]
        if pad is not None:
            import torch
            r = torch.zeros((len(embeddings), pad, embeddings[0].shape[-1]))
            for i, e in enumerate(embeddings):
                r[i, :min(e.shape[0]-1, pad), :] = e[min(e.shape[0]-1, pad)]
            embeddings = r
        return embeddings

    def embed_batch(self, sentences: List, batch_size=64, pad = None):
        """
        Embed large data batchwise.

        Same as embed method but computing the data batchwise to prevent OOM.

        :param sentences: List of sentences.
        :param batch_size: Size of the batches
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: if pad is None a list of embeddings (with varying lengths) is returned, is pad is set a tensor of (num_sentences, pad, embedding_size) will be returned.

        """
        ds = _temporal_dataset(sentences)
        r = []
        for b in tqdm(DataLoader(ds, batch_size=batch_size)):
            r.append(self.embed(b, pad=pad))
        if pad is not None:
            from torch import cat
            r = cat(r,0)
        return r

    def embed_batch_iterator(self, sentences: List, batch_size=64, pad = None):
        """
        Embed large data batchwise as an iterator

        Same as the embed_batch function but as iterator, in case you want to process every batch embeddings before moving on to the next.
        (like writing to disk for very large data)
        :param sentences: List of sentences.
        :param batch_size: Size of the batches
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: Yields the embeddings of a batch per iteration
        """
        ds = _temporal_dataset(sentences,)
        for b in tqdm(DataLoader(ds, batch_size=batch_size)):
            yield self.embed(b, pad=pad)
