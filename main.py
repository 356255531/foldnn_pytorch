import os

from utils import parse_args
args = parse_args()
if args.device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_idx
import torch

from dataset import get_dataloaders
from model import TextCNN
from train_eval import train_single_epoch, evaluate

if args.device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tensor_tpy = {'float': torch.cuda.FloatTensor, 'long': torch.cuda.LongTensor}
else:
    tensor_tpy = {'float': torch.FloatTensor, 'long': torch.LongTensor}

EMB_DIM = 100
SEED = 2019
torch.manual_seed(SEED)


def train_text_cnn(argv=None):
    # Load dataset
    train_dl, valid_dl, test_dl, TEXT, _ = get_dataloaders(SEED, args)

    # Create net
    filter_sizes = [int(i) for i in args.filter_sizes.split(',')]
    num_vocab = len(TEXT.vocab)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    text_nn = TextCNN(
        TEXT.vocab.vectors, EMB_DIM, args.num_filters,
        filter_sizes, args.num_classes, args.dropout, pad_idx).type(tensor_tpy['float'])

    # replace the vocab
    pretrained_embeddings = TEXT.vocab.vectors
    text_nn.embedding.weight.data.copy_(pretrained_embeddings)

    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

    text_nn.embedding.embedding.data[unk_idx] = torch.zeros(EMB_DIM)
    text_nn.embedding.embedding.data[pad_idx] = torch.zeros(EMB_DIM)

    # setup loss and optimizer
    loss_func = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(text_nn.parameters(), lr=args.lr)

    # Start train
    for epoch in range(args.epoch):
        train_single_epoch(text_nn, loss_func, train_dl, opt, epoch, args.num_classes, tensor_tpy)
        evaluate(text_nn, loss_func, test_dl, epoch, args.num_classes, tensor_tpy)


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
