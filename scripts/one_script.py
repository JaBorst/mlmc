import sys
sys.path.insert(0,"/tmp/tmp/pycharm_project_765")
from joblib.test.test_parallel import test_invalid_batch_size
from tqdm import tqdm
import torch
import ignite
from torchcrf import CRF
from seqeval.metrics import precision_score, recall_score,f1_score, classification_report
from mlmc.helpers import charindex
from mlmc.layers import LSTM
import mlmc
import torch
from flair.nn import LockedDropout, WordDropout
import os
weights, vocabulary = mlmc.helpers.load_static(embedding="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-500k.vec")
data = mlmc.data.get_dataset("conll2003en", mlmc.data.SequenceDataset, target_dtype=torch._cast_Long)
os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Sequence2SequenceAbstract(torch.nn.Module):
    def __init__(self, optimizer, optimizer_params = {"lr": 1.0}, device="cpu", **kwargs):
        super(Sequence2SequenceAbstract,self).__init__(**kwargs)

        self.device = device
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params

    def build(self):
        self.to(self.device)
        self.optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

    def get_mask(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            return ((x.unsqueeze(-1)!=0).int().sum(-1)==1).squeeze()

    def length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask.to(self.device)

    def predict(self, sentence):
        self.eval()
        reverse_classes = {v: k for k, v in self.classes.items()}
        single = False
        if isinstance(sentence, str):
            single = 1
            sentence = [sentence, sentence]
        prediction = self(x=self.transform(sentence))
        if single:
            prediction = [prediction[0]]

        labels_predictions = [[reverse_classes[x] for x in sentence] for sentence in prediction]
        return labels_predictions

    def evaluate(self, data, report=False):
        """
        Evaluation, return accuracy and loss
        """
        self.eval()  # set mode to evaluation to disable dropout

        reverse_classes =  {v: k for k, v in self.classes.items()}
        truth = []
        predictions = []
        average = ignite.metrics.Average()

        data_loader = torch.utils.data.DataLoader(data, batch_size=100)
        for i, b in enumerate(data_loader):

            loss, l = self(*self.transform(b["text"], b["labels"]))
            average.update(loss.item())

            labels_predictions = [[reverse_classes[x] for x in sentence] for sentence in l]
            labels_truth = [x.split(" ") for x in b["labels"]]

            truth.extend(labels_truth)
            predictions.extend(labels_predictions)

        self.train()
        return {
            "eval_loss": round(average.compute().item(),4),
            "precision": round(precision_score(labels_truth, labels_predictions),4),
            "recall": round(recall_score(labels_truth, labels_predictions),4),
            "f1": round(f1_score(labels_truth, labels_predictions),4),
            "report": classification_report(labels_truth,labels_predictions) if report else None
        }

    def fit(self, train, valid = None, epochs=1, batch_size=16):
        for e in range(epochs):
            losses = {"loss": str(0.)}
            average = ignite.metrics.Average()
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

            with tqdm(train_loader,
                      postfix=[losses], desc="Epoch %i/%i" %(e+1,epochs)) as pbar:
                for i, b in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    l = self(*self.transform(b["text"], b["labels"]))
                    average.update(l.item())
                    pbar.postfix[0]["loss"] = round(average.compute().item(),6)
                    l.backward()
                    torch.nn.utils.clip_grad_value_(self.parameters(), 5.0)
                    self.optimizer.step()
                    pbar.update()
                pbar.postfix[0]["train"] = self.evaluate(train)
                pbar.update()
                if valid is not None:
                    pbar.postfix[0].update(self.evaluate(valid))
                    pbar.update()




import string
class BILSTMCNN(Sequence2SequenceAbstract):
    def __init__(self, vocabulary, weights,  classes, word_length=45, alphabet=string.ascii_letters+string.punctuation+"0123456789",
                 dropout: float = 0.0,
                 word_dropout: float = 0.05,
                 locked_dropout: float = 0.5
                 , **kwargs):
        super(BILSTMCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.word_length=word_length
        self.alphabet = alphabet
        self.filters=50
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout
        self.sorted_batches=False
        self.kernel_sizes=[3,4]
        self.c_embedding_dim = 30
        self.lstm_hidden = 275

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = LockedDropout(locked_dropout)

        self.w_embedding = torch.nn.Embedding(*weights.shape)
        self.w_embedding.from_pretrained(torch.FloatTensor(weights), freeze=True)

        self.c_embedding = torch.nn.Embedding(len(self.alphabet)+2, self.c_embedding_dim)
        # self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.c_embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.char_lstm=torch.nn.LSTM(input_size=self.c_embedding_dim,hidden_size=50,batch_first=True,bidirectional=True)

        h0 = torch.zeros(2, 1, self.lstm_hidden)
        c0 = torch.zeros(2, 1, self.lstm_hidden)
        torch.nn.init.xavier_normal_(h0, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(c0, gain=torch.nn.init.calculate_gain('relu'))
        self.h0 = torch.nn.Parameter(h0, requires_grad=True)  # Parameter() to update weights
        self.c0 = torch.nn.Parameter(c0, requires_grad=True)
        self.lstm = mlmc.layers.LSTM(weights.shape[1]+len(self.kernel_sizes)*self.filters, self.lstm_hidden, dropout=0.0,
                                       bidirectional=True, batch_first=True)
        self.projection = torch.nn.Linear(in_features=2*self.lstm_hidden, out_features=self.n_classes, )
        self.crf = CRF(len(self.classes), batch_first=True)
        self.build()

    def do_char_convs(self,x):
            return torch.cat(
                [torch.cat(
                    [torch.nn.functional.relu(conv(x[:, k, :, :])).max(-1)[0].unsqueeze(1) for k in range(x.shape[1])], dim=1)
                 for conv in self.convs],
                dim=-1)

    def do_char_lstm(self,x):
        output = []
        for i in range(x.shape[1]):
            o = self.char_lstm(x[:, i, :, :])[1][0]
            output.append(torch.cat([o[0],o[1]],dim=-1))
        return torch.stack(output,dim=1)

    def forward(self, x, truth=None):
        mask = self.length_to_mask(x[2],max_len=x[0].shape[1]).squeeze()

        #character embeddings
        c_embed = self.c_embedding(x[1])
        if self.use_dropout > 0.0: c_embed = self.dropout(c_embed)
        # char_embeddings = self.do_char_convs(c_embed.transpose(-1,-2))
        char_embeddings = self.do_char_lstm(c_embed)
        #Word Embeddings
        w_embed = self.w_embedding(x[0])

        #Combined Embeddings
        embedding = torch.cat([w_embed, char_embeddings], dim=-1)
        if self.use_dropout > 0.0: embedding = self.dropout(embedding)

        packed_embedding = torch.nn.utils.rnn.pack_padded_sequence(embedding, x[2], batch_first=True, enforce_sorted=self.sorted_batches)
        r, _ = self.lstm(packed_embedding, (self.h0.repeat(1, x[0].shape[0], 1),self.c0.repeat(1, x[0].shape[0], 1)))
        r, _ = torch.nn.utils.rnn.pad_packed_sequence(r, batch_first=True)

        # r = self.lstm(embedding, (self.h0.repeat(1, x[0].shape[0], 1), self.c0.repeat(1, x[0].shape[0], 1)))

        if self.use_dropout > 0.0: r = self.dropout(r)

        emissions = self.projection(r)
        if self.training and truth is not None:
            return -self.crf(emissions, truth, mask=mask, reduction="mean")
        elif not self.training and truth is not None:
            return -self.crf(emissions, truth, mask=mask, reduction="mean"), self.crf.decode(emissions, mask=mask)
        else:
            return self.crf.decode(emissions, mask=mask)


    def transform(self, x, y=None, **kwargs):

        length = torch.LongTensor([len(s.split()) for s in x])
        if self.sorted_batches:
            order = torch.argsort(length,descending=True)
            x = [x[k] for k in order]
            y = [y[k] for k in order]

        words = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token,
                                 self.vocabulary["<UNK_TOKEN>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0).to(self.device)
        chars = torch.LongTensor(charindex(x, max([len(s.split()) for s in x]), self.word_length, self.alphabet)).to(
            self.device)

        if y is not None:
            labels = torch.nn.utils.rnn.pad_sequence(
                [torch.LongTensor([self.classes[token] for token in sentence.split(" ")]) for sentence in y],
                batch_first=True, padding_value=0).to(self.device)
            return (words, chars, length), labels
        else:
            return (words, chars, length)


data["classes"] = dict(zip(data["classes"], range(1,len(data["classes"])+1)))
data["classes"]["0"]=0
blc = BILSTMCNN(vocabulary, weights, data["classes"],optimizer=torch.optim.SGD,
                            optimizer_params={"lr": 0.01}, device=device,
                            locked_dropout=0.0 ,word_dropout=0.00, dropout=0.65
                            )

blc.fit(data["train"], data["valid"], epochs=100, batch_size=10)
blc.evaluate(data["test"])

# for b in torch.utils.data.DataLoader(data["train"], batch_size=15): break
# blc(*blc.transform(b["text"], b["labels"]))