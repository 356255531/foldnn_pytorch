import pickle
import tqdm
import os
import pdb

from utils import parse_args
args = parse_args()
if args.device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_idx
import torch

from dataset import TrainDataset
from model import TextCNN
from utils import int_2_onehot

if args.device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    float_t = torch.cuda.FloatTensor
    long_t = torch.cuda.LongTensor
else:
    float_t = torch.FloatTensor
    long_t = torch.LongTensor

filter_sizes = [int(i) for i in args.filter_sizes.split(',')]

torch.manual_seed(0)


def train_text_cnn(argv=None):
    with open(args.google_vocab_mat_file, 'rb') as f:
        google_vocab_mat = pickle.load(f)
    vocab_mat_t = torch.from_numpy(google_vocab_mat)
    vocab_mat_t.type(float_t)

    ds = TrainDataset(args.train_file)
    dl = torch.utils.data.DataLoader(ds, shuffle=True, num_workers=4, batch_size=args.batch_size)

    text_cnn = TextCNN(
        vocab_mat_t, args.emb_dim, args.num_filters,
        filter_sizes, args.dropout, args.num_classes).type(float_t)
    loss_func = torch.nn.BCEWithLogitsLoss()
    params = [x for x in text_cnn.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    for epoch in range(args.epoch):
        pbar = tqdm.tqdm(dl, total=len(dl))
        for batch in pbar:
            x_t, y_t = batch
            x_t = x_t.type(long_t)
            y_t = y_t.type(long_t)
            y_onehot_t = int_2_onehot(args.num_classes, y_t, long_t).type(float_t)

            opt.zero_grad()
            pred = text_cnn(x_t)
            loss = loss_func(pred, y_onehot_t)
            loss.backward()
            opt.step()

            loss = float(loss.detach().cpu().numpy())
            pbar.set_description('loss: {:.4f}'.format(loss))


def simple_test():
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear(2, 1)

        def forward(self, x):
            print(next(self.fc.parameters()).grad)
            x = x.unsqueeze(2)
            x = x.squeeze(2)
            pdb.set_trace()
            return self.fc(x)
    n = Net()
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)
    opt = torch.optim.Adam(n.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    for i in range(100):
        pred = n(x)
        loss = loss_func(pred, y)
        opt.zero_grad()
        loss.backward()
        print(next(n.parameters()).grad)
        print(loss.item())
        opt.step()
# def test_logicnn(argv=None):
#     with open(args.google_vocab_mat_file, 'rb') as f:
#         google_vocab_mat = pickle.load(f)
#     with open(args.positive_data_file, 'rb') as f:
#         train_pos = pickle.load(f)
#         train_pos = np.array(train_pos)
#     with open(args.negative_data_file, 'rb') as f:
#         train_neg = pickle.load(f)
#         train_neg = np.array(train_neg)
#     with open(args.dev_file, 'rb') as f:
#         x_dev, y_dev = pickle.load(f)
#
#     _t_vocab_mat = torch.Tensor(google_vocab_mat).type(float_t)
#     feat = torch.Tensor(train_pos[:10]).type(int_t)
#     text_cnn = TextCNN(_t_vocab_mat, args.emb_dim, args.num_filters, filter_sizes, args.dropout, args.num_classes).type(float_t)
#     logicnn = LogicCnn(text_cnn, 1)
#     y = torch.ones((10,)).type(int_t)
#     print(logicnn.get(feat, y, 1))
#
#
#
# def train_textcnn(argv=None):
#     pass
#
#
# def train_logicnn(argv=None):
#     with open(args.google_vocab_mat_file, 'rb') as f:
#         google_vocab_mat = pickle.load(f)
#     with open(args.positive_data_file, 'rb') as f:
#         train_pos = pickle.load(f)
#         train_pos = np.array(train_pos)
#     with open(args.negative_data_file, 'rb') as f:
#         train_neg = pickle.load(f)
#         train_neg = np.array(train_neg)
#     with open(args.dev_file, 'rb') as f:
#         x_dev, y_dev = pickle.load(f)
#
#     _t_vocab_mat = torch.Tensor(google_vocab_mat).type(float_t)
#     feat_pos = torch.Tensor(train_pos[:10]).type(int_t)
#     feat_neg = torch.Tensor(train_neg[:10]).type(int_t)
#     y_pos = torch.ones((train_pos.shape[0],)).type(int_t)
#     y_neg = torch.zeros((train_neg.shape[0],)).type(int_t)
#     feat = torch.cat([feat_pos, feat_neg], dim=0)
#     y = torch.cat([y_pos, y_neg], dim=0)
#     shuffle_idx = torch.randperm(feat.size()[0])
#     feat = feat[shuffle_idx]
#     y = y[shuffle_idx]
#
#     text_cnn = TextCNN(_t_vocab_mat, args.emb_dim, args.num_filters, filter_sizes, args.dropout, args.num_classes).type(float_t)
#     logicnn = LogicCnn(text_cnn, 1)
#
#     params = [x for x in logicnn.parameters()]
#     optimizer = torch.optim.Adam(params, lr=0.01)
#
#     for epoch in range(1000):
#         optimizer.zero_grad()
#         prob, acc, loss = logicnn.get(feat, y, epoch)
#         loss.backward()
#         optimizer.step()
#         print(acc)


if __name__ == '__main__':
    train_text_cnn()
