import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb


class TextCNN(nn.Module):
    def __init__(self, vocab_mat, emb_dim, num_filters, fiter_sizes, dropout, num_classs):
        super(TextCNN, self).__init__()

        self.word_emb = nn.Embedding.from_pretrained(vocab_mat)

        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, emb_dim)) for fs in fiter_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(fiter_sizes) * num_filters, num_classs)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.word_emb(x)  # (N, W, D)
        print(next(self.fc1.parameters()).grad)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # pdb.set_trace()
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        probs = self.sm(self.fc1(x))  # (N, C)
        return probs


class LogicCnn(nn.Module):
    def __init__(self, student_net, rule_lambda, num_classes=2, C=1, pi_decay_factor=0.95, pi_lower_bound=0.):
        super(LogicCnn, self).__init__()

        self.student_net = student_net
        self.rule_lambda = rule_lambda
        self.C = C
        self.pi_decay_factor = pi_decay_factor
        self.pi_lower_bound = pi_lower_bound
        self.num_classes = num_classes
        self.sm = nn.Softmax()

        self.kl_div = torch.nn.KLDivLoss()

    def forward(self, x):
        student_prob = self.student_net(x[:, 1:])
        rule_distribution = self.compute_rule_constraints(x, student_prob)
        distr = student_prob * rule_distribution
        return self.sm(distr)

    def get(self, x, y, epoch):
        student_prob, student_acc, cross_entropy_loss = self.student_net.get(x, y)
        teacher_prob = self(x).detach()
        acc = (torch.argmax(teacher_prob, dim=1) == y).type(float_t).mean()
        kl_div = self.kl_div(student_prob, teacher_prob)
        pi = 1 - np.max((np.power(self.pi_decay_factor, epoch), self.pi_lower_bound))
        loss = (1.0 - pi) * cross_entropy_loss + pi * kl_div
        return teacher_prob, acc, loss

    def compute_rule_constraints(self, x, student_prob):
        coff = self.C * self.rule_lambda
        distr_y0 = coff * x[:, :1].type(float_t) * student_prob[:, :1]
        distr_y1 = coff * x[:, :1].type(float_t) * student_prob[:, 1:2]
        distr = torch.cat([distr_y0, distr_y1], dim=1)
        return torch.exp(distr)
