import torch
from torchtext import data
from torchtext import datasets
import random


def get_dataloaders(seed, args, rules=None):
    TEXT = data.Field(tokenize='spacy', batch_first=True, rules=rules)
    LABEL = data.LabelField(dtype=torch.float)

    if args.dataset == 'IMDB':
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    elif args.dataset == 'SST':
        train_data, test_data = datasets.SST.splits(TEXT, LABEL)
    elif args.dataset == 'TREC':
        train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)
    else:
        raise ValueError('{} dataset not available.'.format(args.dataset))

    train_data, valid_data = train_data.split(random_state=random.seed(seed))

    TEXT.build_vocab(train_data,
                     max_size=args.max_vocab_size,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    train_loader, valid_loader, test_loader = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=args.batch_size,
        device=args.device)

    return train_loader, valid_loader, test_loader, TEXT, LABEL
