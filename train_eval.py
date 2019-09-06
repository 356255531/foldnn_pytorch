import tqdm
import torch
from utils import label_2_onehot
import numpy as np
import pdb


def train_single_epoch(model, loss_func, train_dl, opt, epoch, num_classes, tensor_typ):
    accs = []
    losses = []

    pbar = tqdm.tqdm(train_dl, total=len(train_dl))
    for batch in pbar:
        x_t, y_t = batch
        x_t = x_t.type(tensor_typ['long']).t()
        y_t = y_t.unsqueeze(1).type(tensor_typ['long'])
        y_onehot_t = label_2_onehot(num_classes, y_t, tensor_typ['long']).type(tensor_typ['float'])

        opt.zero_grad()
        pred = model(x_t[:, :50])
        loss = loss_func(pred, y_onehot_t)
        loss.backward()
        opt.step()

        y_hat_t = torch.argmax(pred, dim=1).type(tensor_typ['long'])
        acc = torch.sum(y_hat_t == y_t) / y_hat_t.shape[0]
        acc = float(acc.detach())
        accs.append(acc)

        loss = float(loss.detach())
        losses.append(loss)
        pbar.set_description(
            'epoch: {}, train acc: {:.3f}, train loss: {:.3f}'.format(epoch + 1, np.mean(accs), np.mean(losses)))


def evaluate(model, loss_func, eval_dl, epoch, num_classes, tensor_typ):
    model.eval()

    pbar = tqdm.tqdm(eval_dl, total=len(eval_dl))
    accs = []
    losses = []
    with torch.no_grad:
        for batch in pbar:
            x_t, y_t = batch
            x_t = x_t.type(tensor_typ['long']).t()
            y_t = y_t.unsqueeze(1).type(tensor_typ['long'])
            y_onehot_t = label_2_onehot(num_classes, y_t, tensor_typ['long']).type(tensor_typ['float'])

            pred = model(x_t)
            loss = loss_func(pred, y_onehot_t)
            loss = float(loss.detach())
            losses.append(loss)

            y_hat_t = torch.argmax(pred, dim=1).type(tensor_typ['long'])
            acc = torch.sum(y_hat_t == y_t) / y_hat_t.shape[0]
            acc = float(acc.detach())
            accs.append(acc)

        pbar.set_description(
            'epoch: {}, eval acc: {:.3f}, eval loss: {:.3f}'.format(epoch + 1, np.mean(accs), np.mean(losses)))