"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.representation import get
import mlmc



class ConceptScores(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, static=None, transformer=None, label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScores, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = 4

        # Original
        self.n_classes = len(classes)

        if transformer is not None:
            self.transformer=True
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, output_hidden_states=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers
        elif static is not None:
            self.transformer=False
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, freeze=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]

        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concept_projection = torch.nn.Linear(self.embedding_dim, self.concepts_dim)

        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad():
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
            else:
                embeddings = self.embedding(x)
        cp = self.concept_projection(embeddings)
        scores = torch.relu(torch.tanh(torch.matmul(cp, self.concepts.permute(1, 0)).mean(-2)))
        output = self.output_projection(scores)
        if return_scores:
            return output, scores
        return output


class ConceptScoresCNN(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, static=None, transformer=None, label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresCNN, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = 4
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5

        # Original
        self.n_classes = len(classes)

        if transformer is not None:
            self.transformer=True
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, output_hidden_states=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers
        elif static is not None:
            self.transformer=False
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, freeze=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]


        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.dynpool = torch.nn.MaxPool2d(( 1, self.window))


        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad=False
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concept_projection = torch.nn.Linear(
            (self.max_len-1)// self.window,#self.filters*( (sum([(self.max_len-s+1)// self.window  for s in self.kernel_sizes]))),
            self.concepts_dim
        )

        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad():
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
            else:
                embeddings = self.embedding(x)
        c = [torch.nn.functional.relu(self.dynpool(conv(embeddings))) for conv in self.convs]
        concat = self.dropout(torch.cat(c, 1))#.view(x.shape[0], -1))
        cp = self.concept_projection(concat)
        concept_scores = torch.relu(torch.tanh(torch.matmul(cp, self.concepts.permute(1, 0)).mean(-2)))
        output = self.output_projection(concept_scores)
        if return_scores:
            return output, concept_scores
        return output


class ConceptScoresCNN(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, static=None, transformer=None, label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresCNN, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = 4
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5

        # Original
        self.n_classes = len(classes)

        if transformer is not None:
            self.transformer=True
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, output_hidden_states=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers
        elif static is not None:
            self.transformer=False
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, freeze=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]



        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.dynpool = torch.nn.MaxPool2d(( 1, self.window))


        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad=False

        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concept_projection = torch.nn.Linear(
            (self.max_len-1)// self.window,#self.filters*( (sum([(self.max_len-s+1)// self.window  for s in self.kernel_sizes]))),
            self.concepts_dim
        )

        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad():
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
            else:
                embeddings = self.embedding(x)

        c = [torch.nn.functional.relu(self.dynpool(conv(embeddings.permute(0,2,1)))) for conv in self.convs]
        concat = self.dropout(torch.cat(c, 1))#.view(x.shape[0], -1))
        cp = self.concept_projection(concat)
        concept_scores = torch.relu(torch.tanh(torch.matmul(cp, self.concepts.permute(1, 0)).mean(-2)))
        output = self.output_projection(concept_scores)
        if return_scores:
            return output, concept_scores
        return output


class ConceptScoresCNNAttention(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, static=None, transformer=None, label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresCNNAttention, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = 4
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5

        # Original
        self.n_classes = len(classes)


        if transformer is not None:
            self.transformer=True
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, output_hidden_states=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers
        elif static is not None:
            self.transformer=False
            self.embedding, self.tokenizer = get(static=static, transformer=transformer, freeze=True)
            self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]

        import math
        self.scaling= torch.nn.Parameter(torch.FloatTensor([1./math.sqrt(self.embedding_dim)]))
        self.scaling.requires_grad=False

        self.attention_linear_query = torch.nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim//2
        )
        self.attention_linear_key = torch.nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim//2
        )


        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.dynpool = torch.nn.MaxPool2d(( 1, self.window))

        # self.concepts = torch.FloatTensor(label_embed)
        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad=True


        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concept_projection = torch.nn.Linear(
            (self.max_len-1)// self.window,#self.filters*( (sum([(self.max_len-s+1)// self.window  for s in self.kernel_sizes]))),
            self.concepts_dim
        )
        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad()  :
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1)
            else:
                embeddings = self.embedding(x)

        self_att = torch.softmax(
            self.scaling*torch.matmul(
                self.attention_linear_query(embeddings),
                self.attention_linear_key(embeddings).permute(0,2,1)).sum(-1),
            dim = -1)

        c = [torch.nn.functional.relu(self.dynpool(conv(embeddings.permute(0,2,1)))) for conv in self.convs]
        concat = self.dropout(torch.cat(c, 1))  # .view(x.shape[0], -1))
        cp = self.concept_projection(concat)
        concept_att = torch.einsum('ijk,lk->ijl',
                                                cp*self.scaling,
                                                self.concepts*self.scaling[0])

        concept_att = F.sigmoid(concept_att)

        concept_scores = (self_att[:,:,None]*concept_att).sum(-2)
        output = self.output_projection(concept_scores)
        if return_scores:
            return output, concept_scores, self_att
        return output



