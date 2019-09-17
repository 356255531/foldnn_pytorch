import torch
from torch.nn import CrossEntropyLoss, KLDivLoss


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def logic_categorical_accuracy(student_prob, teacher_prob, y):
    student_acc = categorical_accuracy(student_prob, y)
    teacher_acc = categorical_accuracy(teacher_prob, y)
    return student_acc, teacher_acc


class LogicLoss(object):
    def __init__(self, pi_decay_factor, pi_lower_bound):
        self.ce_loss_func = CrossEntropyLoss()
        self.kl_div = KLDivLoss()
        self.pi_decay_factor = pi_decay_factor
        self.pi_lower_bound = pi_lower_bound

    def __call__(self, student_prob, teacher_prob, y, epoch):

        teacher_prob = teacher_prob.detach()
        ce_loss = self.ce_loss_func(student_prob, y)
        kl_div = self.kl_div(student_prob, teacher_prob)
        pi = 1 - max(self.pi_decay_factor ** int(epoch), self.pi_lower_bound)
        loss = (1.0 - pi) * ce_loss + pi * kl_div
        return loss
