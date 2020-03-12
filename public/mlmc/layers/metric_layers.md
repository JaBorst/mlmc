Module mlmc.layers.metric_layers
================================

Classes
-------

`Bilinear(dim)`
:   Experimental feature A simple weighted vector sum.
    Idea:
        x[i, j] every activation i of the input corresponds to a distribution over the output class j
        The distributions are weighted and summed.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x, y)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `regularize(self)`
    :

`Metric(dim)`
:   Experimental feature A simple weighted vector sum.
    Idea:
        x[i, j] every activation i of the input corresponds to a distribution over the output class j
        The distributions are weighted and summed.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x, y)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `regularize(self)`
    :