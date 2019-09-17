import os

from utils import parse_args
args = parse_args()
if args.device == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_idx
import torch

from torchtext.rules import but_rule
from dataset import get_dataloaders
from model import TextCNN, LogiCNN
from train_eval import train_single_epoch, train_logic_single_epoch, evaluate, evaluate_logic
from metrics import logic_categorical_accuracy, categorical_accuracy, LogicLoss

if args.device == 'cuda':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 2019
torch.manual_seed(SEED)


def train_text_cnn(argv=None):
    # Load dataset
    train_dl, valid_dl, test_dl, TEXT, _ = get_dataloaders(SEED, args)

    # Create net
    filter_sizes = [int(i) for i in args.filter_sizes.split(',')]
    num_vocab = len(TEXT.vocab)
    EMB_DIM = 100
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    output_dim = 2
    print('Dictionary size: {}'.format(num_vocab))
    text_cnn = TextCNN(num_vocab, EMB_DIM, args.num_filters,
                      filter_sizes, output_dim, args.dropout_r,
                      pad_idx).to(args.device)

    # Load the pretrained_embedding
    pretrained_embeddings = TEXT.vocab.vectors
    text_cnn.embedding.weight.data.copy_(pretrained_embeddings)

    # Init unknown words and pad words embedding
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    text_cnn.embedding.weight.data[unk_idx] = torch.zeros(EMB_DIM)
    text_cnn.embedding.weight.data[pad_idx] = torch.zeros(EMB_DIM)
    text_cnn.embedding.requires_grad = False

    # setup loss and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    acc_func = categorical_accuracy
    opt = torch.optim.Adam(text_cnn.parameters(), lr=args.lr)

    # Start train
    for epoch in range(args.epoch):
        train_single_epoch(text_cnn, loss_func, acc_func, train_dl, opt, epoch)
        evaluate(text_cnn, loss_func, acc_func, test_dl, epoch)


def train_logicnn(argv=None):
    train_dl, valid_dl, test_dl, TEXT, _ = get_dataloaders(SEED, args, rules=[but_rule])

    # Create net
    filter_sizes = [int(i) for i in args.filter_sizes.split(',')]
    num_vocab = len(TEXT.vocab)
    EMB_DIM = 100
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    output_dim = 2
    text_cnn = TextCNN(num_vocab, EMB_DIM, args.num_filters,
                      filter_sizes, output_dim, args.dropout_r,
                      pad_idx).to(args.device)

    # Load the pretrained_embedding
    pretrained_embeddings = TEXT.vocab.vectors
    text_cnn.embedding.weight.data.copy_(pretrained_embeddings)
    text_cnn.embedding.requires_grad = False

    # Init unknown words and pad words embedding
    unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
    text_cnn.embedding.weight.data[unk_idx] = torch.zeros(EMB_DIM)
    text_cnn.embedding.weight.data[pad_idx] = torch.zeros(EMB_DIM)
    text_cnn.embedding.requires_grad = False
    logicnn = LogiCNN(text_cnn, 1)

    # setup loss and optimizer
    loss_func = LogicLoss(args.pi_decay_factor, args.pi_lower_bound)
    acc_func = logic_categorical_accuracy
    opt = torch.optim.Adam(logicnn.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        train_logic_single_epoch(logicnn, loss_func, acc_func, train_dl, opt, epoch)
        evaluate_logic(logicnn, loss_func, acc_func, test_dl, epoch)


if __name__ == '__main__':
    train_text_cnn()
