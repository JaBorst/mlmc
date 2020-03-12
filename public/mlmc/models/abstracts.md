Module mlmc.models.abstracts
============================

Classes
-------

`TextClassificationAbstract(loss=torch.nn.modules.loss.BCEWithLogitsLoss, optimizer=torch.optim.adam.Adam, optimizer_params={'lr': 5e-05}, device='cpu', **kwargs)`
:   Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input
    
    Abstract initializer of a Text Classification network.
    Args:
        loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss)
        optimizer:  One of toch.optim (default: torch.optim.Adam)
        optimizer_params: A dictionary of optimizer parameters
        device: torch device, destination of training (cpu or cuda:0)

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * mlmc.models.KimCNN.KimCNN
    * mlmc.models.LSAN.LSANOriginal
    * mlmc.models.LSANOriginalTransformer.LSANOriginalTransformer
    * mlmc.models.LSAN_reimplementation.LabelSpecificAttention
    * mlmc.models.SKG_ML.SKG
    * mlmc.models.XMLCNN.XMLCNN
    * mlmc.models.ZAGCNN.ZAGCNN
    * mlmc.models.ZAGCNNLM.ZAGCNNLM

    ### Methods

    `build(self)`
    :   Internal build method.

    `evaluate(self, data, batch_size=50, return_roc=False, return_report=False, mask=None)`
    :   Evaluation, return accuracy and loss and some multilabel measure
        
        Returns p@1, p@3, p@5, AUC, loss, Accuracy@0.5, Accuracy@mcut, ROC Values, class-wise F1, Precision and Recall.
        Args:
            data: A MultilabelDataset with the data for evaluation
            batch_size: The batch size of the evaluation loop. (Larger is often faster, but it should be small enough to fit into GPU memory. In general it can be larger than batch_size in training.
            return_roc: If True, the return dictionary contains the ROC values.
            return_report: If True, the return dictionary will contain a class wise report of F1, Precision and Recall.
        Returns:
            A dictionary with the evaluation measurements.

    `evaluate_classes(self, classes_subset=None, **kwargs)`
    :   wrapper for evaluation function if you just want to evaluate on subsets of the classes.

    `fit(self, train, valid=None, epochs=1, batch_size=16, valid_batch_size=50, classes_subset=None)`
    :   Training function
        
        Args:
            train: MultilabelDataset used as training data
            valid: MultilabelDataset to keep track of generalization
            epochs: Number of epochs (times to iterate the train data)
            batch_size: Number of instances in one batch.
            valid_batch_size: Number of instances in one batch  of validation.
        Returns:
            A history dictionary with the loss and the validation evaluation measurements.

    `num_params(self)`
    :   Count the number of trainable parameters.
        
        Returns:
            The number of trainable parameters

    `predict(self, x, return_scores=False, tr=0.5, method='hard')`
    :   Classify sentence string  or a list of strings.
        
        Args:
            x:  A list of the text instances.
            return_scores:  If True, the labels are returned with their corresponding confidence scores
            tr: The threshold at which the labels are returned.
            method: Method of thresholding
                    (hard will cutoff at ``tr``, mcut will look for the largest distance in
                    confidence between two labels following each other and will return everything above)
        
        Returns:
            A list of the labels

    `predict_dataset(self, data, batch_size=50, tr=0.5, method='hard')`
    :   Predict all labels for a dataset int the mlmc.data.MultilabelDataset format.
        
        For detailed information on the arcuments see `mlmc.models.TextclassificationAbstract.predict`
        
        Args:
            data: A MultilabelDataset
            batch_size: Batch size
            tr: Threshold
            method: mcut or hard
        
        Returns:
            A list of labels

    `threshold(self, x, tr=0.5, method='hard')`
    :   Thresholding function for outputs of the neural network.
        So far a hard threshold ( tr=0.5, method="hard")  is supported and
        dynamic cutting (method="mcut")
        
        Args:
            x: A tensor
            tr: Threshold
            method: mcut or hard
        
        Returns:

    `transform(self, x)`
    :   A standard transformation function from text to network input format
        
        The function looks for the tokenizer attribute. If it doesn't exist the transform function has to
        be implemented in the child class
        
        Args:
            x: A list of text
        
        Returns:
            A tensor in the network input format.