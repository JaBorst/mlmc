Module mlmc.representation.representations
==========================================
Loading Embeddings and Word embeddings in an automated fashion.

Functions
---------

    
`get(model, **kwargs)`
:   Universal get function for text embedding methods and tokenizers.
    
    Args:
        model: Model name (one of [ glove50, glove100, glove200, glove300] or any of the models on https://huggingface.co/models
        **kwargs:  Additional arguments in case of transformers. for example ``output_hidden_states=True`` for returning hidden states of transformer models.
            For details on the parameters for the specific models see https://huggingface.co
    
    
    Returns:
         A tuple of embedding and corresponding tokenizer
    
    Examples:
        ```
        embedder, tokenizer = get("bert-base-uncased")
        embedding = embedder(tokenizer("A sentence of various words"))
        ```
    The variable ``embedding`` will contain a torch tensor of shape
    
     (1, sequence_length, embedding_dim)

    
`get_embedding(name, **kwargs)`
:   Load a static word embedding from file.
    Args:
        name: File name of the word embedding. (Expects a text file in the glove format)
    Returns: A tuple of embedding and corresponding tokenizer

    
`get_transformer(model='bert', **kwargs)`
:   Get function for transformer models
    Args:
        model: Model name
        **kwargs: Additional keyword arguments
    
    Returns:  A tuple of embedding and corresponding tokenizer

    
`is_transformer(name)`
:   A check function. True if the ``name`` argument if found to be a valid transformer model name.
    
    Args:
        name: model name (see get)
    
    Returns: bool

    
`load_static(embedding)`
:   Load the embedding from a testfile.
    
    Args:
        embedding: one of [glove50, glove100, glove200, glove300]
    
    Returns: The embedding matrix and the vocabulary.

    
`map_vocab(query, vocab, maxlen)`
:   Map a query ( a list of lists of tokens ) to indices using the vocab mapping and
    pad (or cut respectively) all to maxlen.
    Args:
        query: a list of lists of tokens
        vocab: A mapping from tokens to indices
        maxlen: Maximum lengths of the lists
    
    Returns: A torch.Tensor with shape (len(query), maxlen)