Module mlmc.layers.lstm
=======================

Classes
-------

`LSTM(*args, dropouti=0.0, dropoutw=0.0, dropouto=0.0, batch_first=True, unit_forget_bias=True, **kwargs)`
:   A copy of the better LSTM github repository
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.rnn.LSTM
    * torch.nn.modules.rnn.RNNBase
    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, input, hx=None)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`LSTMRD(input_size, hidden_size, bias=True, batch_first=False, dropout=0.5, recurrent_dropout=0.5, bidirectional=False)`
:   A self implementation of the LSTM with a tensorflow like recurrent dropout on the activations of the
    reccurrent cell
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x, hidden=None)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`VariationalDropout(dropout, batch_first=False)`
:   Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

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