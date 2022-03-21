import torch

from mlmc.models.abstracts.abstracts_multi_output import TextClassificationAbstractMultiOutput
from mlmc.modules.module_KimCNN import KimCNNModule
from mlmc.representation import get


class MoKimCNN(TextClassificationAbstractMultiOutput):
    """
    Implementation of Yoon Kim 2014 KimCNN Classification network for Multilabel Application
    (added support for Language Models).
    """

    def __init__(self, mode="transformer", kernel_sizes=(3, 4, 5, 6), filters=100, dropout=0.5, **kwargs):
        """
        Class constructor and initialization of every hyperparameter

        :param mode:  One of (trainable, untrainable, multichannel, transformer).
                    Transformer has to be used in combination with representation being a transformer model name
                     (see: https://huggingface.co/transformers/pretrained_models.html).
                    In combination with the other three one of the glove embeddings can be used
                    (glove50, glove100, glove200, glove300)
                    'trainable' will finetune the word embedding used, whereas "untrainable" will freeze the
                    embedding layer.
                    'multichannel' will combine two embedding layers, one for finetuning on the task, one frozen.
        :param kernel_sizes: Sizes of the kernel used for the convolution
        :param filters: Number of filters used in the convolution
        :param dropout: Dropout rate
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstractMultiOutput`
        """
        super(MoKimCNN, self).__init__( **kwargs)

        self._config["kernel_sizes"] = kernel_sizes
        self._config["filters"] = filters
        self._config["dropout"] = dropout
        self._config["mode"] = mode

        self.modes = ("trainable", "untrainable", "multichannel", "transformer")
        assert self._config["mode"]  in self.modes, f"{self._config['mode'] } not in ({self.modes})"

        self.l = 2 if  self._config["mode"] == "multichannel" else 1

        if self._config["mode"]  == "multichannel":
            self.embedding_channel2, self.tokenizer_channel2 = get(model=self.representation, freeze=not self.finetune)

            # Layers
        self.kimcnn_module = KimCNNModule(in_features=self.embeddings_dim, kernel_sizes= self._config["kernel_sizes"],
                                          filters=self._config["filters"], dropout=self._config["dropout"])
        self.dropout_layer = torch.nn.Dropout(self._config["dropout"])
        self.projection = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.l * self.kimcnn_module.out_features, out_features=x) for x in
             self.n_classes])

        self.build()

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        e = self.embed_input(x)
        c = self.kimcnn_module(e.permute(0, 2, 1))
        if self._config["mode"] == "multichannel":
            e2 = self.embedding_channel2(x)
            c = torch.cat([c, self.kimcnn_module(e2.permute(0, 2, 1))], -1)
        output = [x(self.dropout_layer(c)) for x in self.projection]
        return output
