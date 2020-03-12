Module mlmc.representation.embedder
===================================

Classes
-------

`Embedder(representation, method='first_token', device='cpu', return_device='cpu')`
:   A class for embedding lists of sentences.
    
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

    ### Methods

    `embed(self, sentences, pad=None)`
    :   Embedding method for a list of sentences.
        :param sentences:  List of sentences
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: if pad is None a list of embeddings (with varying lengths) is returned, is pad is set a tensor of (num_sentences, pad, embedding_size) will be returned.

    `embed_batch(self, sentences, batch_size=64, pad=None)`
    :   Embed large data batchwise.
        
        Same as embed method but computing the data batchwise to prevent OOM.
        
        :param sentences: List of sentences.
        :param batch_size: Size of the batches
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: if pad is None a list of embeddings (with varying lengths) is returned, is pad is set a tensor of (num_sentences, pad, embedding_size) will be returned.

    `embed_batch_iterator(self, sentences, batch_size=64, pad=None)`
    :   Embed large data batchwise as an iterator
        
        Same as the embed_batch function but as iterator, in case you want to process every batch embeddings before moving on to the next.
        (like writing to disk for very large data)
        :param sentences: List of sentences.
        :param batch_size: Size of the batches
        :param pad: (default: None) If pad is set all sentences will be padded (or cut repectively) to the desired length.
        :return: Yields the embeddings of a batch per iteration