Module mlmc.graph.graph_loaders
===============================

Functions
---------

    
`load_NELL()`
:   Download and return the NELL Graph (Never Ending Lanuage Learning
    Will return only elements that have a confidence of 0.999 and larger.
    :return: A networkx.DiGraph

    
`load_wordnet()`
:   Loading the wordnet graph as a networkx.DiGraph
    :return: A networkx.DiGraph containing wordnet.

    
`load_wordnet_sample(num=1000)`
:   Loading the wordnet graph as a networkx.DiGraph
    :num: The size of the subsample ( Only the first n synsets will be added )
    :return: A networkx.DiGraph containing wordnet.