import torch


##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class KimCNNModule(torch.nn.Module):
    """
    Implementation of Yoon Kim 2014 KimCNN Classification network for Multilabel Application (added support for Language Models).
    """

    def __init__(self, in_features, kernel_sizes=(3, 4, 5, 6), filters=100, dropout=0.5):
        """Class constructor and intialization of every hyperparameters

        :param classes:  A dictionary of the class label and the corresponding index
        :param mode:  One of (trainable, untrainable, multichannel, transformer).
                    Transformer has to be used in combination with representation being a transformer model name (see: https://huggingface.co/transformers/pretrained_models.html).
                    In combination with the other three one of the glove embeddings can be used (glove50, glove100, glove200, glove300)
                    'trainable' will finetune the wordembedding used, whereas "untrainable" will freeze the embedding layer.
                    'multichannel' will combine two embedding layers, one for finetuning on the task, one frozen.
        :param representation: The name of the representation to use. glove* or one of the hugginface transformers models.
        :param kernel_sizes: Sizes of the kernel used for the convolution
        :param filters: Number of filters used in the convolution
        :param dropout: Droupout rate
        :param max_len: Maximum length input sequences. Longer sequences will be cut.
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.TextclassificationAbstract`
         """
        super(KimCNNModule, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.in_features = in_features
        self.filters = filters
        self.dropout = dropout
        self.out_features = len(self.kernel_sizes) * self.filters
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.in_features, self.filters, k) for k in self.kernel_sizes])

    def forward(self, x):
        c = [torch.nn.functional.relu(conv(x).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        c = torch.cat(c, 1)
        return c
