Module mlmc.models.LSAN
=======================
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py

Classes
-------

`LSANOriginal(classes, representation, label_embed=None, label_freeze=True, lstm_hid_dim=300, d_a=200, max_len=500, **kwargs)`
:   https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    
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

    `init_hidden(self, size)`
    :