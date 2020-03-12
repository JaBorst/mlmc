Module mlmc.layers.mogrifier
============================

Classes
-------

`MogLSTM(input_sz, hidden_sz, mog_iterations)`
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

    `forward(self, x, init_states=None)`
    :   Assumes x is of shape (batch, sequence, feature)

    `init_weights(self)`
    :

    `mogrify(self, xt, ht)`
    :

`MogrifierLSTM(input_size, hidden_size, mogrify_steps=2, learn_initial_states=False, batch_first=True)`
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

`MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)`
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

    `forward(self, x, states)`
    :   inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)

    `init_parameters(self)`
    :

    `mogrify(self, x, h)`
    :