import torch


class LabelAttention(torch.nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim):
        super(LabelAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=1, kdim=self.hidden_dim, vdim=self.hidden_dim)

        self.projection = torch.nn.Linear(self.input_dim, self.hidden_dim)

        self.label_repr = torch.nn.Parameter(torch.Tensor(n_classes, self.hidden_dim))
        torch.nn.init.kaiming_normal_(self.label_repr)

    def forward(self, x):
        ls =  self.label_repr.unsqueeze(0).repeat(x.shape[0],1,1).permute((1, 0, 2))
        output, att = self.attention( ls, self.projection(x).permute((1, 0, 2)), self.projection(x).permute((1, 0, 2)))
        return output.permute((1, 0, 2)), att

class LabelSpecificSelfAttention(torch.nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim):
        super(LabelSpecificSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.to_hidden = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.to_label = torch.nn.Parameter(torch.Tensor(self.hidden_dim, self.n_classes))

        torch.nn.init.kaiming_normal_(self.to_hidden)
        torch.nn.init.kaiming_normal_(self.to_label)

    def forward(self,x):
        att = torch.softmax(torch.matmul(torch.tanh(torch.matmul(x, self.to_hidden)), self.to_label), -1)
        return torch.matmul(att.permute(0,2,1),x), att

class AdaptiveCombination(torch.nn.Module):
    def __init__(self, input_dim, n_classes ):
        super(AdaptiveCombination, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.alpha_weights = torch.nn.Parameter(torch.Tensor(input_dim,1))
        self.beta_weights = torch.nn.Parameter(torch.Tensor(input_dim,1))
        torch.nn.init.kaiming_normal_(self.alpha_weights)
        torch.nn.init.kaiming_normal_(self.beta_weights)

    def forward(self, x):
        alpha = torch.sigmoid(torch.matmul(x[0], self.alpha_weights))
        beta = torch.sigmoid(torch.matmul(x[1], self.beta_weights))

        #constrain the sum to one
        alpha = alpha / (alpha + beta)
        beta = beta / (alpha + beta)
        output = alpha*x[0] + beta*x[1]
        return output

#
# ac = LabelSpecificSelfAttention(n_classes=10, input_dim=300, hidden_dim=150)
# ac(torch.randn(2,140,300))[1].shape
# # la = LabelAttention(10, 200, 300)
# i = torch.randn(2,140,200)
# la(i)[0].shape