Module mlmc.models.ZAGCNN
=========================
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)

Classes
-------

`ZAGCNN(classes, adjacency, representation='roberta', max_len=300, dropout=0.5, **kwargs)`
:   Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input
    
    Abstract initializer of a Text Classification network.
    Args:
        loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss)
        optimizer:  One of toch.optim (default: torch.optim.Adam)
        optimizer_params: A dictionary of optimizer parameters
        device: torch device, destination of training (cpu or cuda:0)

    ### Ancestors (in MRO)

    * mlmc.models.abstracts.TextClassificationAbstract
    * torch.nn.modules.module.Module

    ### Methods

    `create_labels(self, classes)`
    :

    `forward(self, x)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `set_adjacency(self, adj)`
    :