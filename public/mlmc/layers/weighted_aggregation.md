Module mlmc.layers.weighted_aggregation
=======================================

Classes
-------

`AttentionWeightedAggregation(in_features, d_a)`
:   Aggregate a tensor weighted by weights obtained from self attention
    https://arxiv.org/pdf/1703.03130.pdf
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x, aggr=None, return_att=True)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.