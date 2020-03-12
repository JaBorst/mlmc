Module mlmc.metrics.multilabel
==============================

Classes
-------

`AUC_ROC(n_classes, reduction='macro')`
:   Multilabel iterative AUC_ROC. Ignite API like

    ### Methods

    `compute(self)`
    :

    `reset(self)`
    :

    `update(self, batch)`
    :

`MultiLabelReport(classes, check_zeros=False)`
:   Multilabel iterative F1/Precision/Recall. Ignite API like

    ### Methods

    `compute(self)`
    :

    `update(self, batch)`
    :