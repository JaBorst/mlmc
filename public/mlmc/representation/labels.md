Module mlmc.representation.labels
=================================

Functions
---------

    
`makemultilabels(query, maxlen, tagset=None)`
:   

    
`makesequencelabels(query, maxlen, tagset)`
:   

    
`one_hot(...)`
:   one_hot(tensor, num_classes=-1) -> LongTensor
    
    Takes LongTensor with index values of shape ``(*)`` and returns a tensor
    of shape ``(*, num_classes)`` that have zeros everywhere except where the
    index of last dimension matches the corresponding value of the input tensor,
    in which case it will be 1.
    
    See also `One-hot on Wikipedia`_ .
    
    .. _One-hot on Wikipedia:
        https://en.wikipedia.org/wiki/One-hot
    
    Arguments:
        tensor (LongTensor): class values of any shape.
        num_classes (int):  Total number of classes. If set to -1, the number
            of classes will be inferred as one greater than the largest class
            value in the input tensor.
    
    Returns:
        LongTensor that has one more dimension with 1 values at the
        index of last dimension indicated by the input, and 0 everywhere
        else.
    
    Examples:
        >>> F.one_hot(torch.arange(0, 5) % 3)
        tensor([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]])
        >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
        tensor([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0]])
        >>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
        tensor([[[1, 0, 0],
                 [0, 1, 0]],
                [[0, 0, 1],
                 [1, 0, 0]],
                [[0, 1, 0],
                 [0, 0, 1]]])

    
`schemetransformer(column, scheme='BIOES', multilabel=False)`
:   

    
`to_scheme(tagset, scheme='iobes', outside='O')`
: