Module mlmc.models.lm_mogrifier
===============================

Classes
-------

`MogrifierLMCharacter(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~1234567890', hidden_size=128, emb_dim=50, n_layers=1, mogrify_steps=2, learn_initial_states=True, max_len=128, dropout=0.5, inter_dropout=0.2, **kwargs)`
:   Abstract class for Language Models. Defines fit, evaluate, predict and threshold methods for virtually any
    language model.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * mlmc.models.abstracts_lm.LanguageModelAbstract
    * torch.nn.modules.module.Module

    ### Methods

    `decode(self, s)`
    :

    `encode(self, s)`
    :

    `forward(self, x, representations=False)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `transform(self, s)`
    :

`MogrifierLMWord(word_list, hidden_size=128, emb_dim=50, n_layers=1, mogrify_steps=2, learn_initial_states=True, max_len=128, dropout=0.5, inter_dropout=0.2, **kwargs)`
:   Abstract class for Language Models. Defines fit, evaluate, predict and threshold methods for virtually any
    language model.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * mlmc.models.abstracts_lm.LanguageModelAbstract
    * torch.nn.modules.module.Module

    ### Methods

    `decode(self, s)`
    :

    `encode(self, s)`
    :

    `forward(self, x, representations=False)`
    :   Defines the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `transform(self, s)`
    :