import torch

from mlmc.models.abstracts.abstracts import TextClassificationAbstract
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.graph import get as graph_get
import networkx as nx
from mlmc.data import MultiLabelDataset
from tqdm import tqdm
import random


class TextClassificationAbstractGraph(TextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
    Abstract class for Multilabel Models. Defines fit, evaluate, predict and threshold methods for virtually any
    multilabel training.
    This class is not meant to be used directly.
    Also provides a few default functions:
        _init_input_representations(): if self.representations exists, the default will load a embedding and corresponding tokenizer
        transform(): If self.tokenizer exists the default method wil use this to transform text into the models input

    """
    def __init__(self, graph, embed="label", **kwargs):
        """
        Abstract initializer of a Text Classification network.
        Args:
            target: single label oder multilabel mode. defined by keystrings: ("single", "multi"). Sets some basic options, like loss function, activation and
                    metrics to sensible defaults.
            activation: The activation function applied to the output. Only used for metrics and when you want to return scores in predict. (default: torch.softmax for "single", torch.sigmoid for "multi")
            loss: One of the torch.nn  losses (default: torch.nn.BCEWithLogitsLoss for "multi" and torch.nn.CrossEntropyLoss for "single")
            optimizer:  One of toch.optim (default: torch.optim.Adam)
            optimizer_params: A dictionary of optimizer parameters
            device: torch device, destination of training (cpu or cuda:0)
        """
        super(TextClassificationAbstractGraph,self).__init__(**kwargs)

        self.embed=embed
        if isinstance(graph, str) or isinstance(graph, list):
            self.graph = graph
            self.kb = graph_get(graph)
            self.dynamic_graph=True
        elif isinstance(graph, nx.Graph):
            self.kb = graph
            self.dynamic_graph=False
        self.no_nodes =len(self.kb)



    def complete_score_graph(self,x, n=20, method="hard"):
        x = [x] if isinstance(x, str) else x
        scores = self.act(self(self.transform(x), return_graph_scores=True))
        from copy import deepcopy
        new_graph = deepcopy(self.label_subgraph)
        for sentence, sentence_scores, ones in zip(x, scores, self.threshold(scores, method=method)):
            keywords = str([list(self.label_subgraph.nodes)[i] for score, i in zip(*sentence_scores.topk(n))])

            new_graph.add_node(sentence,
                               prediction={list(self.classes)[i]: sentence_scores[i].cpu().item() for i in torch.where(ones[:self.n_classes]==1)[0]},
                               type="text",
                               keywords=keywords)


            for score, i in zip(*sentence_scores.topk(n)):
                new_graph.add_edge(sentence,list(self.label_subgraph.nodes)[i], weight=score.cpu().item(), type="label")
        return new_graph


    def score_graph(self,x, n=20, method="mcut"):
        x = [x] if isinstance(x, str) else x
        scores = self.act(self(self.transform(x), return_graph_scores=True))
        new_graph = nx.OrderedDiGraph()
        for sentence, sentence_scores, ones in zip(x, scores, self.threshold(scores, method=method)):
            keywords = str([list(self.label_subgraph.nodes)[i] for score, i in zip(*sentence_scores.topk(n)) if i > self.n_classes])

            new_graph.add_node(sentence,
                               prediction={list(self.classes)[i]: sentence_scores[i].cpu().item() for i in
                                           torch.where(ones[:self.n_classes] == 1)[0]},
                               type="text",
                               keywords=keywords)
            for score, i in zip(*sentence_scores.topk(n)):
                new_graph.add_edge(sentence, list(self.label_subgraph.nodes)[i], weight=score.cpu().item(),
                                   type="label")
        return new_graph


    def fit_graphsubsample(self, train, valid, epochs=1, negative=1., batch_size=16, valid_batch_size=50, classes_subset=None, patience=-1, tolerance=1e-2,
            return_roc=False):
        history = []
        evaluation = []
        zeroshot_classes = list(set(valid.classes.keys()) - set(train.classes.keys()))
        print("Found Zero-shot Classes: ", str(zeroshot_classes))

        import datetime
        id = str(hash(datetime.datetime.now()))[1:7]
        from ..data import SingleLabelDataset
        if isinstance(train, SingleLabelDataset) and self.target != "single":
            print("You are using the model in multi mode but input is SingeleLabelDataset.")
            return 0

        validation = []
        train_history = {"loss": []}

        assert not (type(
            train) == SingleLabelDataset and self.target == "multi"), "You inserted a SingleLabelDataset but chose multi as target."
        assert not (type(
            train) == MultiLabelDataset and self.target == "single"), "You inserted a MultiLabelDataset but chose single as target."

        best_loss = 10000000
        last_best_loss_update = 0
        classes_backup = self.classes.copy()
        label_subgraph_backup = self.label_subgraph.copy()
        from ignite.metrics import Average
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" % (e + 1, epochs), ncols=100) as pbar:
                for i, b in enumerate(train_loader):

                    self.optimizer.zero_grad()


                    self.classes=classes_backup.copy()
                    self.label_subgraph = label_subgraph_backup.copy()
                    y = self.subsample(b["labels"].to(self.device), negative=negative)


                    x = self.transform(b["text"])
                    output = self(x)
                    if hasattr(self, "regularize"):
                        l = self.loss(output, y) + self.regularize()
                    else:
                        l = self.loss(output, y)
                    if self.use_amp:
                        with amp.scale_loss(l, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        l.backward()

                    self.optimizer.step()
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(), 2 * self.PRECISION_DIGITS)
                    pbar.update()
                # EVALUATION
                self.create_labels(classes_backup)
                validation.append(self.evaluate_classes(classes_subset=classes_subset,
                                                        data=valid,
                                                        batch_size=valid_batch_size,
                                                        return_report=True,
                                                        return_roc=return_roc))
                printable = {
                    "overall": {"micro": validation[-1]["report"]["micro avg"],
                                "macro": validation[-1]["report"]["macro avg"]},
                    "zeroshot": {
                        x: validation[-1]["report"][x] for x in zeroshot_classes
                    }

                }

                pbar.postfix[0].update(printable)
                pbar.update()
                if patience > -1:
                    if valid is None:
                        print("check validation loss")
                        if best_loss - average.compute().item() > tolerance:
                            print("update validation and checkoint")
                            best_loss = average.compute().item()
                            torch.save(self.state_dict(), id + "_checkpoint.pt")
                            # save states
                            last_best_loss_update = 0
                        else:
                            print("increment no epochs")
                            last_best_loss_update += 1

                        if last_best_loss_update >= patience:
                            print("breaking at %i" % (patience,))
                            print("Early Stopping.")
                            break
                    elif valid is not None:
                        if best_loss - validation[-1]["valid_loss"] > tolerance:
                            best_loss = validation[-1]["valid_loss"]
                            torch.save(self.state_dict(), id + "_checkpoint.pt")
                            # save states
                            last_best_loss_update = 0
                        else:
                            last_best_loss_update += 1

                        if last_best_loss_update >= patience:
                            print("Early Stopping.")
                            break

            train_history["loss"].append(average.compute().item())
        if patience > -1:
            self.load_state_dict(torch.load(id + "_checkpoint.pt"))

        return {"train": history, "valid": evaluation}

    def subsample(self, batch, negative=1.0):
        occurring_labels = [list(self.classes.keys())[x] for x in list(set(torch.where(batch == 1)[1].tolist()))]

        remaining_classes = list(set(self.classes.keys()) - set(occurring_labels))
        negative_classes = random.sample(remaining_classes, min(int(negative*len(occurring_labels)),len(remaining_classes)) )

        current_classes = sorted(occurring_labels+negative_classes)

        subgraph = nx.OrderedGraph()
        subgraph.add_nodes_from(current_classes)
        for n in current_classes:
            neighbours =[x for x in self.label_subgraph[n]]
            neighbours = random.sample(neighbours,int(len(neighbours)/2))
            subgraph.add_nodes_from( neighbours)
        nx.set_node_attributes(subgraph, dict([x for x in self.label_subgraph.nodes(True) if x in subgraph]))

        current_classes = dict(zip(current_classes,range(len(current_classes))))
        mapping = {self.classes[k]:v  for k,v in current_classes.items()}

        self.label_subgraph=subgraph
        self.classes = current_classes

        new_ind = [[mapping[m.item()] for m in torch.where(x==1)[0]] for x in batch]
        batch = torch.stack([ torch.nn.functional.one_hot(torch.LongTensor(labels), len(self.classes)).sum(0) for labels in new_ind]).float().to(self.device)


        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.label_subgraph).toarray()).float()

        tmp_adj = tmp_adj + torch.eye(tmp_adj.shape[0])
        tmp_adj[tmp_adj != 0] = 1
        self.adfdense= tmp_adj
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)

        if self.embed=="label":
            self.label_embeddings = self.tokenizer(list(self.label_subgraph.nodes), pad=True, maxlen=10).to(self.device)
        else:
            embedsequences = [x[1][self.embed] if self.embed in x[1].keys() and x[1][self.embed] != "" else x[0] for x in self.label_subgraph.nodes(True)]
            p = max([len(x) for x in embedsequences])
            self.label_embeddings = self.tokenizer(embedsequences, pad=True, maxlen=min(int(p/3),512)).to(self.device)

        self.label_embeddings_dim = self.embeddings_dim
        self.no_nodes = len(self.label_subgraph)
        self.n_classes = len(current_classes)
        return batch