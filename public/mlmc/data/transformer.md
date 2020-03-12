Module mlmc.data.transformer
============================
A collection of frequently used tranforming functions for strings or tensors

Functions
---------

    
`clean(x)`
:   Remove every character in a string that is not ascii, punctuation or whitespace
    :param x:
    :return:

    
`label_smoothing(x, alpha=0.1)`
:   Take one hot vector and return a smoothed version of it. ( [0,0,1] -> [0.05, 0.05, 0.9])
    :param x: one hot vector
    :param alpha: Amount of the smoothing
    :return: smoothed one hot vector

    
`label_smoothing_random(x, alpha=0.1)`
:   Label smoothing with noise.
    :param x: One hot vector
    :param alpha: Amount of the smoothing
    :return: A smoothed one hot vector with added noise.