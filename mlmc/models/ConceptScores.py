"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.representation import get, is_transformer
import mlmc



class ConceptScores(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, layers=4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScores, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.representation=representation
        self._init_input_representations()

        # Original
        self.n_classes = len(classes)
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
    def __init__(self, classes, layers=4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresCNN, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5
        self.representation=representation
        self._init_input_representations()

        # Original
        self.n_classes = len(classes)

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
    def __init__(self, classes, layers=4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresCNN, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5

        self.representation=representation
        self._init_input_representations()

        # Original
        self.n_classes = len(classes)

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
    def __init__(self, classes, layers = 4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresCNNAttention, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5
        self.representation=representation
        self._init_input_representations()

        # Original
        self.n_classes = len(classes)

        import math
        self.query_projection = torch.nn.Linear(self.embedding_dim, 512)
        self.key_projection = torch.nn.Linear(self.embedding_dim, 512)
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
                self.query_projection(embeddings),
                self.key_projection(embeddings).permute(0,2,1)).sum(-1), dim = -1) #/self.scaling.to(self.device), dim=-1)

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



class ConceptScoresAttention(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, layers = 4,representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresAttention, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100
        self.window=5

        self.representation=representation
        self._init_input_representations()


        # Original
        self.n_classes = len(classes)
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad= not label_freeze


        self.self_att = torch.nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=2,dropout=0.5)
        self.concept_att = torch.nn.MultiheadAttention(embed_dim= 100, num_heads=2,dropout=0.5)

        self.importance = torch.nn.Linear(in_features=self.max_len, out_features=1)

        self.concept_projection = torch.nn.Linear(
            self.embedding_dim,
            self.concepts_dim
        )
        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        # self.concept_embedding_output_projection = torch.nn.Linear(self.concepts_dim, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad()  :
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1).permute(1,0,2)
            else:
                embeddings = self.embedding(x)

        embeddings, attention_scores = self.self_att(embeddings, embeddings, embeddings)
        word_importance = torch.softmax(self.importance(attention_scores), dim=1)

        concepts, concept_attention = self.concept_att(
            query=self.concept_projection(embeddings),
            key=self.concepts.unsqueeze(1).repeat(1,x.shape[0],1),
            value=self.concepts.unsqueeze(1).repeat(1,x.shape[0],1))

        concept_scores = torch.sigmoid((word_importance*concept_attention).sum(1))
        # concepts_mean = (word_importance*concepts.permute(1,0,2)).sum(1)
        # concept_mean_prediction = self.concept_embedding_output_projection(concepts_mean)
        output = self.output_projection(concept_scores)
        if return_scores:
            return output, concept_attention, word_importance
        return output


class ConceptScoresRelevanceWithImportanceWeights(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, layers = 4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresRelevanceWithImportanceWeights, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 256
        self.window=5


        self.representation=representation
        self._init_input_representations()

        # Original
        self.n_classes = len(classes)
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad= not label_freeze
        self.self_att = torch.nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=2, dropout=0.5)

        # self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])

        self.importance = torch.nn.Linear(in_features=self.embedding_dim, out_features=1)
        self.metric = torch.nn.Parameter(torch.zeros((self.concepts_dim, self.concepts_dim)))
        torch.nn.init.xavier_uniform_(self.metric)

        self.concept_projection_up = torch.nn.Linear(
            self.embedding_dim,
            self.concepts_dim
        )
        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        # self.concept_embedding_output_projection = torch.nn.Linear(self.concepts_dim, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad()  :
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1).permute(1,0,2)
            else:
                embeddings = self.embedding(x).permute(1,0,2)
        embeddings, attention_scores = self.self_att(embeddings, embeddings, embeddings)
        embeddings=embeddings.permute(1,0,2)
        concept_scores = torch.matmul(torch.matmul(self.concept_projection_up(embeddings), self.metric), self.concepts.transpose(1,0))
        concept_scores = torch.tanh(concept_scores) /concept_scores.norm(dim=-1, keepdim=True)


        imp = torch.softmax(self.importance(embeddings).squeeze(), dim=-1).unsqueeze(-1)
        weighted_sum = (imp*concept_scores).sum(-2)
        output = self.output_projection(weighted_sum)
        if return_scores:
            return output, concept_scores, imp.squeeze()
        return output


class ConceptScoresRelevance(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, layers = 4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScoresRelevance, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 256
        self.window=5


        self.representation=representation
        self._init_input_representations()

        # Original
        self.n_classes = len(classes)
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad= not label_freeze
        self.self_att = torch.nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=2, dropout=0.5)

        # self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])


        self.metric = torch.nn.Parameter(torch.zeros((self.concepts_dim, self.concepts_dim)))
        torch.nn.init.xavier_uniform_(self.metric)

        self.concept_projection_up = torch.nn.Linear(
            self.embedding_dim,
            self.concepts_dim
        )
        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        # self.concept_embedding_output_projection = torch.nn.Linear(self.concepts_dim, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad()  :
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1).permute(1,0,2)
            else:
                embeddings = self.embedding(x).permute(1,0,2)

        embeddings, attention_scores = self.self_att(embeddings, embeddings, embeddings)
        embeddings=embeddings.permute(1,0,2)
        concept_scores = torch.matmul(torch.matmul(self.concept_projection_up(embeddings), self.metric), self.concepts.transpose(1,0))
        concept_scores = torch.relu(concept_scores) /concept_scores.norm(dim=-1, keepdim=True)
        concept_scores = concept_scores.mean(-2)
        output = self.output_projection(concept_scores)
        if return_scores:
            return output, concept_scores
        return output



class KimCNN2Branch(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, layers = 4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(KimCNN2Branch, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 256
        self.window=5
        self.representation = representation

        # Original
        self.n_classes = len(classes)
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self._init_input_representations()
        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad=not label_freeze

        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])

        self.projection = torch.nn.Linear(in_features=len(self.kernel_sizes) * self.filters,
                                          out_features=self.n_classes)

        self.concept_projection_up = torch.nn.Linear(
            self.embedding_dim,
            self.concepts_dim
        )
        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad():
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1).permute(0,2,1)
            else:
                embeddings = self.embedding(x).permute(0,2,1)

        c = torch.cat([torch.nn.functional.relu(conv(embeddings).permute(0, 2, 1).max(1)[0]) for conv in self.convs], -1)
        prediction1 = self.projection(c)

        embeddings = embeddings.permute(0, 2, 1)
        concept_scores = torch.matmul(self.concept_projection_up(embeddings),
                         self.concepts.transpose(1, 0)).mean(-2) / embeddings.shape[-2]
        prediction2 = self.output_projection (concept_scores)

        output = prediction1 + prediction2
        if return_scores:
            return output, concept_scores
        return output


class ConceptProjection(TextClassificationAbstract):
    def __init__(self, classes, layers = 4, representation="roberta", label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptProjection, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 256
        self.window=5
        self.representation = representation
        self.n_classes = len(classes)

        # Original
        self._init_input_representations()

        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts.requires_grad=not label_freeze

        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.projection_to_concepts = torch.nn.Linear(in_features=len(self.kernel_sizes) * self.filters,
                                          out_features=self.n_concepts)
        self.projection_concepts_to_classes = torch.nn.Linear(in_features=self.concepts_dim,
                                          out_features=self.n_classes)
        self.concept_projection_up = torch.nn.Linear(
            self.embedding_dim,
            self.concepts_dim
        )
        self.dropout= torch.nn.Dropout(0.5)
        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            if self.transformer:
                embeddings = torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1).permute(0,2,1)
            else:
                embeddings = self.embedding(x).permute(0,2,1)
        c = torch.cat([torch.nn.functional.relu(conv(embeddings).permute(0, 2, 1).max(1)[0]) for conv in self.convs], -1)
        concept_scores = torch.softmax(self.projection_to_concepts(c),dim=-1)
        concept_vector = torch.matmul(concept_scores, self.concepts)
        prediction2 = self.projection_concepts_to_classes(concept_vector)
        output = prediction2
        if return_scores:
            return output, concept_scores
        return output