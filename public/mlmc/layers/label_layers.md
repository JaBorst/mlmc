Module mlmc.layers.label_layers
===============================
Layers that are concerned with scoring labels

Classes
-------

`AdaptiveCombination(input_dim, n_classes)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
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

`LabelAttention(n_classes, input_dim, hidden_dim, label_repr=None, freeze=True)`
:   Reimplementation of label attention as described in paper: Label-specific Document representations.
    Might be deprecated.
    
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

`LabelEmbeddingAttention(n_classes, input_dim, hidden_dim, label_embedding)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
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

`LabelEmbeddingScoring(n_classes, input_dim, label_repr, similarity='cosine', label_freeze=True)`
:   Layer that keeps a representation (static Embedding) and compares the input to all vectors. The metric
    should be freely choosable
    
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

`LabelSpecificSelfAttention(n_classes, input_dim, hidden_dim)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
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