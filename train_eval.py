import tqdm
import torch
import numpy as np
import pdb


def train_single_epoch(model, loss_func, acc_func, train_dl, opt, epoch):
    accs = []
    losses = []

    model.train()

    pbar = tqdm.tqdm(train_dl, total=len(train_dl))
    for batch in pbar:
        x, y = batch
        y = y.long()
        preds = model(x)
        loss = loss_func(preds, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = acc_func(preds, y)
        acc = float(acc.detach())
        accs.append(acc)

        loss = float(loss.detach())
        losses.append(loss)
        pbar.set_description(
            'epoch: {}, train acc: {:.3f}, train loss: {:.3f}'.format(epoch + 1, np.mean(accs), np.mean(losses)))


def train_logic_single_epoch(model, loss_func, acc_func, train_dl, opt, epoch):
    student_accs = []
    teacher_accs = []
    losses = []

    model.train()

    pbar = tqdm.tqdm(train_dl, total=len(train_dl))
    for batch in pbar:
        x, y = batch.text, batch.label
        but_x, if_but = batch.but_text, batch.if_but
        y = y.long()
        if_but = if_but.long()
        import pdb
        pdb.set_trace()
        student_prob, teacher_prob = model(x, [if_but], [but_x])
        loss = loss_func(student_prob, teacher_prob, y, epoch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        student_acc, teacher_acc = acc_func(student_prob, teacher_prob, y)
        student_acc, teacher_acc = float(student_acc.detach()), float(teacher_acc.detach())
        student_accs.append(student_acc)
        teacher_accs.append(teacher_acc)

        loss = float(loss.detach())
        losses.append(loss)

        pbar.set_description(
            'epoch: {}, train student acc: {:.3f}, train teacher acc: {:.3f}, train loss: {:.3f}'.format(
                epoch + 1, np.mean(student_accs), np.mean(teacher_accs), np.mean(losses)))


def evaluate(model, loss_func, acc_func, eval_dl, epoch):
    model.eval()

    accs = []
    losses = []

    with torch.no_grad():
        pbar = tqdm.tqdm(eval_dl, total=len(eval_dl))
        for batch in pbar:
            x, y = batch
            y = y.long()
            preds = model(x)
            loss = loss_func(preds, y)
            loss = float(loss.detach())
            losses.append(loss)

            acc = acc_func(preds, y)
            acc = float(acc.detach())
            accs.append(acc)

            pbar.set_description(
                'epoch: {}, eval acc: {:.3f}, eval loss: {:.3f}'.format(epoch + 1, np.mean(accs), np.mean(losses)))


def evaluate_logic(model, loss_func, acc_func, eval_dl, epoch):
    model.eval()

    teacher_accs = []
    student_accs = []
    losses = []

    with torch.no_grad():
        pbar = tqdm.tqdm(eval_dl, total=len(eval_dl))
        for batch in pbar:
            x, y = batch
            y = y.long()
            student_prob, teacher_prob = model(x)
            loss = loss_func(student_prob, teacher_prob, y, epoch)
            loss = float(loss.detach())
            losses.append(loss)

            student_acc, teacher_acc = acc_func(student_prob, teacher_prob, y)
            student_acc, teacher_acc = float(student_acc.detach()), float(teacher_acc.detach())
            student_accs.append(student_acc)
            teacher_accs.append(teacher_acc)

            pbar.set_description(
                'epoch: {}, eval student acc: {:.3f}, eval teacher acc: {:.3f}, loss: {:.3f}'.format(epoch + 1, np.mean(student_accs), np.mean(teacher_accs), np.mean(losses)))