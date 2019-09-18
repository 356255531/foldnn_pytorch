import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        fc = self.fc(cat)

        # fc = [batch size, n_classes]

        return fc


class LogiCNN(nn.Module):
    def __init__(self, student_net, rule_lambda, C=1):
        super(LogiCNN, self).__init__()

        self.student_net = student_net
        self.rule_lambda = rule_lambda
        self.C = C

    def forward(self, x, if_rules, rule_xs):
        """
        Now only suitable for A-but-B rule
        :param x:
        :param if_rules:
        :param rule_xs:
        :return:
        """
        student_prob = self.student_net(x)
        rule_distribution = self.compute_but_rule_constraints(if_rules[0], rule_xs[0])
        teacher_prob = student_prob * rule_distribution
        return student_prob, teacher_prob

    def compute_but_rule_constraints(self, if_but, but_xs):
        coff = self.C * self.rule_lambda

        logic_prob = self.student_net(but_xs)
        sigma_pos = logic_prob[:, 1:2]
        logic_truth_val_pos = (1 + sigma_pos) / 2
        logic_truth_val_neg = (2 - sigma_pos) / 2

        distr_neg = coff * if_but.float() * (1 - logic_truth_val_neg)
        distr_pos = coff * if_but.float() * (1 - logic_truth_val_pos)
        distr = torch.cat([distr_neg, distr_pos], dim=1)
        return torch.exp(-distr)
