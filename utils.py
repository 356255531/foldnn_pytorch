import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Image inpaiting.')

    parser.add_argument('-google_vocab_mat_file', type=str, default="./data/google_vocab_mat.pkl")
    parser.add_argument('-train_file', type=str, default="./data/train.pkl")
    parser.add_argument('-test_file', type=str, default="./data/test.pkl")
    parser.add_argument('-dev_file', type=str, default="./data/dev.pkl")
    parser.add_argument('-logic_train_file', type=str, default="./data/logic_train.pkl")
    parser.add_argument('-logic_test_file', type=str, default="./data/logic_test.pkl")
    parser.add_argument('-logic_dev_file', type=str, default="./data/logic_dev.pkl")

    if torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    parser.add_argument('-device', type=str, default=default_device)
    parser.add_argument('-gpu_idx', type=str, default='1')

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=float, default=1028)
    parser.add_argument('-lr', type=float, default=0.01)

    parser.add_argument('-emb_dim', type=int, default=300)
    parser.add_argument('-num_filters', type=int, default=128)
    parser.add_argument('-filter_sizes', type=str, default="3,4,5")
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-num_classes', type=int, default=2)

    args = parser.parse_args()

    return args


def int_2_onehot(num_class, y, dtype):
    y_onehot = torch.zeros((y.shape[0], num_class))
    y_onehot = y_onehot.type(dtype)
    y_onehot.scatter_(1, y, 1)

    return y_onehot


if __name__ == '__main__':
    y = torch.LongTensor([[1], [0]])
    y_onehot = int_2_onehot(2, y)
    print(y_onehot)