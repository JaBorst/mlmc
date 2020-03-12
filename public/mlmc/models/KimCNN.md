Module mlmc.models.KimCNN
=========================

Classes
-------

`KimCNN(classes, mode='transformer', representation='roberta', kernel_sizes=[3, 4, 5, 6], filters=100, dropout=0.5, max_len=500, **kwargs)`
:   Implementation of Yoon Kim 2014 KimCNN Classification network for Multilabel Application (added support for Language Models).
    
    Class constructor and intialization of every hyperparameters
    
    :param classes:  A dictionary of the class label and the corresponding index
    :param mode:  One of (trainable, untrainable, multichannel, transformer).
    Transformer has to be used in combination with representation being a transformer model name (see: https://huggingface.co/transformers/pretrained_models.html).
    In combination with the other three one of the glove embeddings can be used (glove50, glove100, glove200, glove300)
    'trainable' will finetune the wordembedding used, whereas "untrainable" will freeze the embedding layer.
    'multichannel' will combine two embedding layers, one for finetuning on the task, one frozen.
    
    Args:
        representation: The name of the representation to use. glove* or one of the hugginface transformers models.
        kernel_sizes: Sizes of the kernel used for the convolution
        filters: Number of filters used in the convolution
        dropout: Droupout rate
        max_len: Maximum length input sequences. Longer sequences will be cut.
        kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.TextclassificationAbstract`

    ### Ancestors (in MRO)

    * mlmc.models.abstracts.TextClassificationAbstract
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.