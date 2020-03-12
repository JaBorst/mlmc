Module mlmc.loss.smoothing_loss
===============================

Classes
-------

`NoiseSmoothBCEWithLogitsLoss()`
:   Torch like loss function but with smoothed labels. [Confident Learning]
    ToDo:
     - Make the smoothing function an argument
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.loss.BCEWithLogitsLoss
    * torch.nn.modules.loss._Loss
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, inputs, targets)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.