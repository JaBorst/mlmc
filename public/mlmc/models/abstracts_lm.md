Module mlmc.models.abstracts_lm
===============================

Classes
-------

`LanguageModelAbstract(loss=torch.nn.modules.loss.CrossEntropyLoss, optimizer=torch.optim.adam.Adam, optimizer_params={'lr': 5e-05}, device='cpu', **kwargs)`
:   Abstract class for Language Models. Defines fit, evaluate, predict and threshold methods for virtually any
    language model.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * mlmc.models.lm_mogrifier.MogrifierLMCharacter
    * mlmc.models.lm_mogrifier.MogrifierLMWord

    ### Methods

    `build(self)`
    :

    `evaluate(self, data, batch_size=50)`
    :   Evaluation, return accuracy and loss

    `fit(self, train, valid=None, epochs=1, batch_size=16, valid_batch_size=50)`
    :

    `generate(self, prompt='', steps=100, sample=True, beta=1)`
    :

    `num_params(self)`
    :

    `representations(self, s)`
    :

    `transform(self, x)`
    :