from typing import List

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..representation import get, is_transformer
import torch

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
    def __init__(self, representation, method="first_token", device="cpu", return_device="cpu"):
        """
        Holding an embedding to embed lists of sentences with huggingface or glove.

        This class returns an embedding vector for every whitespace separated token in the data.
        (Tested only on all BERT-like models which use a prefixe for whitespaces.)
        For now only the first_token policy is supported.

        ToDo:
            - Support Pooling method
            - Averaging Method

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

        self.methods =  ("first_token", "all", "sentence", "pool")
        assert method in self.methods
        self.set_method(method)

        self.maxlen=500

    def set_method(self, m=None):
        """
        Sets method used for embedding tokens.

        :param m: Method name (first_token, all, sentence, pool)
        :return: True if the method was set successfully, else False
        """
        if m is None:
            print(f"m should be one of the following: {self.methods}")
            return False
        assert m in self.methods, f"m should be one of the following: {self.methods}"

        self.method = m
        return True

    def _get_all(self, sentences, pad):
        """
        TODO: Documentation

        :param sentences: List of sentences.
        :param pad: If pad is set all sentences will be padded (or cut respectively) to the desired length
        :return:
        """
        t = self.tok(x=sentences, return_start=False, maxlen=self.maxlen, pad=True, as_mask=False, add_special_tokens=False)
        with torch.no_grad():
            embeddings = self.emb(t.to(self.device))[0].to(self.return_device)
        embeddings = [e[i] for e, i in zip(embeddings, t !=self.tok.tokenizer.pad_token_id)]
        return self._pad_it(embeddings,pad)

    def _get_first_token(self, sentences, pad):
        """
        TODO: Documentation

        :param sentences: List of sentences.
        :param pad: If pad is set all sentences will be padded (or cut respectively) to the desired length
        :return:
        """
        t, ind = self.tok(x=sentences, return_start=True,maxlen=self.maxlen, pad=True, as_mask=False, add_special_tokens=False)
        with torch.no_grad():
            embeddings = self.emb(t.to(self.device))[0].to(self.return_device)
        embeddings =  [e[i[i < embeddings.shape[1]]] for e, i in zip(embeddings, ind)]
        return self._pad_it(embeddings,pad)

    def _get_sentence(self, sentences, pad):
        """
        TODO: Documentation

        :param sentences: List of sentences.
        :param pad: If pad is set all sentences will be padded (or cut respectively) to the desired length
        :return:
        """
        t = self.tok(x=sentences, return_start=False, maxlen=self.maxlen, pad=True, as_mask=False, add_special_tokens=False)
        with torch.no_grad():
            embeddings = self.emb(t.to(self.device))[1].to(self.return_device)
        return embeddings

    def _get_pool(self, sentences, pad):
        """
        TODO: Documentation

        :param sentences: List of sentences
        :param pad: If pad is set all sentences will be padded (or cut respectively) to the desired length
        :return:
        """
        t, ind = self.tok(x=sentences, return_start=True, maxlen=self.maxlen, pad=True, as_mask=False, add_special_tokens=False)
        with torch.no_grad():
            embeddings = self.emb(t.to(self.device))[0].to(self.return_device)
        embeddings = [torch.stack([e[start:end].max(0)[0] for start,end in zip(i.tolist(), i.tolist()[1:] + [i[-1]+5 if i[-1]+5 <embeddings.shape[1] else None] )]) for e, i in zip(embeddings, ind) ]
        return self._pad_it(embeddings,pad)

    def _pad_it(self, emb: List, pad):
        """
        Pads a tensor.

        :param emb: A three-dimensional tensor
        :param pad: Pads (or cuts) the second dimension of the input tensor to the desired length
        :return: A padded tensor of shape (emb.shape[0], pad, emb.shape[2])
        """
        if pad is not None:
            padded = torch.zeros(len(emb) if isinstance(emb,list) else emb.shape[0], pad, emb[0][0].shape[-1])
            for i, e in enumerate(emb):
                padded[i, :min(e.shape[0], pad), :] = e[:min(e.shape[0], pad)]
            emb = padded
        return emb

    def embed(self, sentences: List, pad=None):
        """
        embedding method for a list of sentences.
        :param sentences:  List of sentences
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: if pad is None a list of embeddings (with varying lengths) is returned, is pad is set a tensor of (num_sentences, pad, embedding_size) will be returned.
        """

        if is_transformer(self.representation):
            if self.method == "all":
                embeddings = self._get_all(sentences, pad=pad)
            elif self.method == "first_token":
                embeddings = self._get_first_token(sentences, pad=pad)
            elif self.method == "sentence":
                embeddings = self._get_sentence(sentences, pad=pad)
            elif self.method == "pool":
                embeddings = self._get_pool(sentences, pad=pad)
        else:
            s_length = [len(x.split()) for x in sentences]
            t = self.tok(sentences, pad if pad is not None else max(s_length))
            with torch.no_grad():
                embeddings = self.emb(t.to(self.device)).to(self.return_device)
            if pad is None:
                embeddings = [e[:i] for e,i in zip(embeddings,s_length)]

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
        for b in DataLoader(ds, batch_size=batch_size):
            yield self.embed(b, pad=pad)
