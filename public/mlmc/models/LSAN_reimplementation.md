Module mlmc.models.LSAN_reimplementation
========================================

Classes
-------

`LabelSpecificAttention(classes, representation, label_embedding=None, max_len=600, dropout=0.5, **kwargs)`
:   Reimplementation of https://github.com/EMNLP2019LSAN/LSAN
    
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

    `forward(self, x)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.