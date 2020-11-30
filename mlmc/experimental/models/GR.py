from ...models.abstracts.abstracts import TextClassificationAbstract
from ...models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
import torch
import networkx as nx
from ...representation import is_transformer
from ...modules import LSANNCModule, LSANGraphModule,DynamicWeightedFusion

class GR(TextClassificationAbstract,TextClassificationAbstractZeroShot):
    def __init__(self, classes, graph, representation, maxlen=200, decision_noise=0.001, **kwargs):
        super(GR, self).__init__(**kwargs)
        # My Stuff
        self.classes = classes
        self.n_layers = 1
        self.representation = representation
        self.graph = graph
        self.max_len = maxlen
        self.decision_noise = decision_noise

        self.adj = torch.Tensor(nx.adjacency_matrix(self.graph).toarray())
        self.adjacency = torch.stack(torch.where(self.adj == 1), dim=0).to(self.device)
        self.n_classes = len(classes)
        self._init_input_representations()

        if not is_transformer(self.representation):
            self.projection_input = torch.nn.LSTM(self.embeddings_dim,
                                                  hidden_size=int(self.embeddings_dim/2),
                                                  num_layers=1,
                                                  batch_first=True,
                                                  bidirectional=True)

        indeg = torch.Tensor([self.graph.in_degree(n) + 1e-10 for n in self.graph.nodes])
        outdeg = torch.Tensor([self.graph.out_degree(n) + 1e-10 for n in self.graph.nodes])
        self.adj = self.adj / outdeg[:,None] / indeg[None,:]

        self.adj = torch.nn.Parameter(self.adj, requires_grad=False)


        self.dropout = torch.nn.Dropout(0.5)

        from mlmc.representation import get as rget
        self.label_embedding, self.label_tokenizer = rget(model="glove300")
        self.label_embedding_dim = 300

        self.node_embeddings = self.label_tokenizer(self.graph.nodes, 5)
        self.create_labels(self.classes)
        self.hidden_features=512
        self.lsanliker = LSANGraphModule(self.embeddings_dim, self.label_embedding_dim,hidden_features=self.hidden_features, noise=self.decision_noise)
        self.lsanr = LSANNCModule(self.embeddings_dim, self.embeddings_dim,hidden_features=self.hidden_features, noise=self.decision_noise)
        self.dwf = DynamicWeightedFusion(self.embeddings_dim, n_inputs=2, noise=self.decision_noise)
        self.output = torch.nn.Linear(self.embeddings_dim*2, 1)
        self.sim = torch.nn.CosineSimilarity(-1)

        self.input_attention_projection = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim)
        self.classes_attention_projection = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim)
        self.build()

    def forward(self, x, return_weights=False):
        input_mask = (x!=0)
        label_mask = (self.classes_labels!=0).to(self.device)
        e = self.embed_input(x)
        classes_labels = self.embed_input(self.classes_labels.to(self.device))

        if not is_transformer(self.representation):
            # In case of word embeddings, create context dependent representation by applying an LSTM layer
            e,_ = self.projection_input(e)
            classes_labels,_ = self.projection_input(classes_labels)

        # input_att = torch.softmax(2*torch.bmm(e,self.input_attention_projection(e).transpose(1,2)).sum(1),-1)
        classes_att = torch.softmax(2*torch.bmm(classes_labels,self.classes_attention_projection(classes_labels).transpose(1,2)).sum(1),-1)

        with torch.no_grad():
            node_mask = (self.node_embeddings != 0).to(self.device)
            node_embeddings = self.label_embedding(self.node_embeddings.to(self.device)).sum(1) / node_mask.sum(1, keepdim=True)

        input_sim = self.lsanliker(x=e, nodes=node_embeddings, graph = self.adjacency, mask=input_mask)
        label_sim = self.lsanliker(x=classes_labels, nodes=node_embeddings,  graph = self.adjacency, mask = label_mask)

        self._prediction_embeddings = input_sim
        self._prediction_labels = label_sim

        input_sim, w2 =  self.dwf([e, input_sim])
        label_sim, w3 = self.dwf([classes_labels,label_sim])

        r, w1 = self.lsanr(input_sim, (classes_att[:, :, None] * label_sim).sum(1), return_weights=True)
        r = self.output(r).squeeze()
        if return_weights:
            return r, w1, w2, w3
        else:
            return r


    def get_weights(self, data, batch_size=64):
        p, *weights = self(self.transform("test").to(gr.device), return_weights=True)
        weight_lists_correct = [[torch.tensor([0.]).to(self.device)] for _ in range(len(weights))]
        weight_lists_incorrect = [[torch.tensor([0.]).to(self.device)] for _ in range(len(weights))]
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        self.eval()
        with torch.no_grad():
            for i, b in enumerate(data_loader):
                p, *weights = self(self.transform(b["text"]).to(self.device), return_weights=True)
                p = self.act(p)
                correct = ((self._threshold_fct(p)).float()== b["labels"].to(p.device)).all(-1)
                for i in range(len(weights)):
                    if correct.any():
                        if weights[i].shape[0] != len(b["text"]):
                            weight_lists_correct[i].append(weights[i][...,1].flatten())
                        else:
                            weight_lists_correct[i].append(weights[i][correct][...,1].flatten())

                    if not correct.all():
                        if weights[i].shape[0] != len(b["text"]):
                            weight_lists_incorrect[i].append(weights[i][..., 1].flatten())
                        else:
                            weight_lists_incorrect[i].append(weights[i][torch.logical_not(correct)][..., 1].flatten())


        self.train()
        weight_lists_correct = [torch.cat(x).cpu() for x in weight_lists_correct]
        weight_lists_incorrect = [torch.cat(x).cpu() for x in weight_lists_incorrect]
        return *weight_lists_correct, *weight_lists_incorrect

    # def label_embed(self, classes):
    #     return self.tokenizer(self.classes.keys(),10)


    def create_labels(self, classes):
        self.classes = classes
        self.classes_labels = self.tokenizer(self.classes.keys(),10)
        self.n_classes = len(classes)

        l = list(classes.items())
        l.sort(key=lambda x: x[1])
        if not hasattr(self, "_trained_classes"):
            self._trained_classes = []
        self._zeroshot_ind = torch.LongTensor([1 if x[0] in self._trained_classes else 0 for x in l])
        self._mixed_shot = not (self._zeroshot_ind.sum() == 0 or self._zeroshot_ind.sum() == self._zeroshot_ind.shape[0]).item()


    def act(self, x):
        if "softmax" in self.activation.__name__ or "softmin" in self.activation.__name__:
            x = self.activation(x, -1)
        else:
            x = self.activation(x)
        # if self._mixed_shot:
        #     return mean_scaling(x, self._zeroshot_ind.to(x.device))
        return x

    def plot_weights(self, nsl_data, zsl_data, gzsl_data, names = [], h = None):
        self.create_labels(gzsl_data.classes)
        gzsl_weights = self.get_weights(gzsl_data)

        self.create_labels(zsl_data.classes)
        zsl_weights = self.get_weights(zsl_data)

        self.create_labels(nsl_data.classes)
        nsl_weights = self.get_weights(nsl_data)

        n_weights = int(len(nsl_weights)/2)
        if n_weights != len(names):
            names = [names[n] if n < len(names) else f"None {n}" for n in range(n_weights)]
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(12, 5))
        [axs[0, 0].axes.hist(x, label=l, bins=100, range=[0., 1.], density=True) for x, l in
         zip(gzsl_weights[:n_weights], names)]
        [axs[1, 0].axes.hist(x, label=l, bins=100, range=[0., 1.], density=True) for x, l in
         zip(gzsl_weights[n_weights:], names)]
        [axs[0, 1].axes.hist(x, label=l, bins=100, range=[0., 1.], density=True) for x, l in
         zip(zsl_weights[:n_weights], names)]
        [axs[1, 1].axes.hist(x, label=l, bins=100, range=[0., 1.], density=True) for x, l in
         zip(zsl_weights[n_weights:], names)]
        [axs[0, 2].axes.hist(x, label=l, bins=100, range=[0., 1.], density=True) for x, l in
         zip(nsl_weights[:n_weights], names)]
        [axs[1, 2].axes.hist(x, label=l, bins=100, range=[0., 1.], density=True) for x, l in
         zip(nsl_weights[n_weights:],names)]
        if h is not None:
            axs[0, 0].set_title(f'gzsl loss={h["valid"][0]["gzsl"]["valid_loss"]}')
            axs[0, 1].set_title(f'zsl loss={h["valid"][0]["zsl"]["valid_loss"]}')
            axs[0, 2].set_title(f'nsl loss={h["valid"][0]["nsl"]["valid_loss"]}')
        else:
            axs[0, 0].set_title(f'gzsl')
            axs[0, 1].set_title(f'zsl')
            axs[0, 2].set_title(f'nsl')
        axs[0,2].legend()
        return fig, axs
