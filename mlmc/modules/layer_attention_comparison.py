import torch


class LabelAttention(torch.nn.Module):
    """Reimplementation of label attention as described in paper: Label-specific Document representations.
    Might be deprecated.
    """

    def __init__(self, n_classes, input_dim, hidden_dim, label_repr=None, freeze=True):
        """
        Class constructor.

        :param n_classes: Number of classes in classes mapping
        :param input_dim: Size of each input sample
        :param hidden_dim: Hidden state size
        :param label_repr: If not None, loads the specified label embeddings, else creates them using Kaiming Initialization
        :param freeze: If true, the label embeddings will be updated in the training process
        """
        super(LabelAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        if self.hidden_dim is not None:
            if self.hidden_dim != self.input_dim:
                self.projection = torch.nn.Linear(self.input_dim, self.hidden_dim)

        if label_repr is None:
            self.label_repr = torch.nn.Parameter(torch.Tensor(n_classes, self.hidden_dim))
            torch.nn.init.kaiming_normal_(self.label_repr)
        else:
            assert label_repr.shape[-1] == hidden_dim, "label embedding dimension must equal hidden_dim"
            self.label_repr = torch.nn.Parameter(label_repr)
            self.label_repr.requires_grad = freeze

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensors. Calculates self-attention and
        label-specific document representation.

        :param x: Input tensor
        :return: Output tensors
        """
        if self.hidden_dim is not None:
            if self.hidden_dim != self.input_dim:
                x = self.projection(x)
        A = torch.softmax(torch.matmul(x, self.label_repr.permute(1, 0)), -1)
        output = torch.matmul(A.permute(0, 2, 1), x)
        return output, A


class LabelEmbeddingAttention(torch.nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim, label_embedding):
        """
        Class constructor.

        :param n_classes: Number of classes in classes mapping
        :param input_dim: Size of each input sample
        :param hidden_dim: Hidden state size
        :param label_embedding: Embeddings of the labels
        """
        super(LabelEmbeddingAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        if self.hidden_dim is not None:
            if self.hidden_dim != self.input_dim:
                self.projection = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.label_repr = label_embedding

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensors. Calculates self-attention and
        label-specific document representation.

        :param x: Input tensor
        :return: Output tensors
        """
        A = torch.softmax(torch.matmul(x, self.label_repr.permute(1, 0)), -1)
        output = torch.matmul(A.permute(0, 2, 1), x)
        return output, A


class LabelEmbeddingScoring(torch.nn.Module):
    """Layer that keeps a representation (static Embedding) and compares the input to all vectors. The metric
        should be freely choosable
    """

    def __init__(self, n_classes, input_dim, label_repr, similarity="cosine", label_freeze=True):
        """
        Class constructor.

        :param n_classes: Number of classes in classes mapping
        :param input_dim: Size of each input sample
        :param label_repr: Embeddings of the labels
        :param similarity: Similarity measure used, either "cosine" or "euclidean"
        :param label_freeze: If true, the label embeddings will not be updated in the training process
        """
        super(LabelEmbeddingScoring, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        assert similarity in ["cosine", "euclidean"], "Distance metric %s not implemented." % (similarity,)
        self.similarity = similarity

        self.label_repr = torch.nn.Parameter(torch.from_numpy(label_repr).float())
        self.label_repr.requires_grad = not label_freeze
        self.projection = torch.nn.Linear(self.input_dim, self.label_repr.shape[-1])

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        x = self.projection(x)
        if self.similarity == "cosine":
            output = torch.matmul(
                x / torch.norm(x, p=2, dim=-1).unsqueeze(-1),
                (self.label_repr / torch.norm(self.label_repr, p=2, dim=-1).unsqueeze(-1)).transpose(0, 1)
            )
        if self.similarity == "euclidean":
            output = torch.sigmoid(
                torch.norm((x.unsqueeze(2) - self.label_repr.unsqueeze(0).unsqueeze(1)), p=2, dim=-1)
            )
        return output


class LabelSpecificSelfAttention(torch.nn.Module):
    """
    Reimplementation of self-attention as described in paper: Label-Specific Document Representation for Multi-Label
    Text Classification.
    """
    def __init__(self, n_classes, input_dim, hidden_dim):
        """
        Class constructor.

        :param n_classes: Number of classes in classes mapping
        :param input_dim: Size of each input sample
        :param hidden_dim: Hidden state dimension
        """
        super(LabelSpecificSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # self.to_hidden = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.to_label = torch.nn.Parameter(torch.Tensor(self.hidden_dim, self.n_classes))

        self.to_hidden = torch.nn.Linear(self.input_dim, self.hidden_dim)
        # self.to_label = torch.nn.Linear(self.hidden_dim, self.n_classes)

        # torch.nn.init.kaiming_normal_(self.to_hidden)
        torch.nn.init.kaiming_normal_(self.to_label)

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor containing label-specific document representation under the self-attention mechanism and
        label-word attention score
        """
        att = torch.softmax(torch.matmul(
            torch.tanh(self.to_hidden(x)), self.to_label),
            -1)
        # att = torch.softmax(self.to_label(torch.tanh(self.to_hidden(x))), -1)
        return torch.matmul(att.permute(0, 2, 1), x), att


class AdaptiveCombination(torch.nn.Module):
    """
    Reimplementation of adaptive attention fusion as described in paper: Label-Specific Document Representation for
    Multi-Label Text Classification.
    """
    def __init__(self, input_dim, n_classes):
        """
        Class constructor.

        :param input_dim: Size of each input sample
        :param n_classes: Number of classes in classes mapping
        """
        super(AdaptiveCombination, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.alpha_weights = torch.nn.Linear(input_dim, 1)
        self.beta_weights = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor containing self-attention and label-attention representations
        :return: Output tensor containing final document representation based on fusion weights
        """
        alpha = torch.sigmoid(self.alpha_weights(x[0]))
        beta = torch.sigmoid(self.beta_weights(x[1]))

        # constrain the sum to one
        alpha = alpha / (alpha + beta)
        beta = beta / (alpha + beta)
        output = alpha * x[0] + beta * x[1]
        return output
