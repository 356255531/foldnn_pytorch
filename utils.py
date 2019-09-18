import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Image inpaiting.')

    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-gpu_idx', type=str, default='1')

    parser.add_argument('-dataset', type=str, default='IMDB')
    parser.add_argument('-max_vocab_size', type=int, default=1_000)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=float, default=64)
    parser.add_argument('-lr', type=float, default=0.01)

    parser.add_argument('-num_filters', type=int, default=100)
    parser.add_argument('-filter_sizes', type=str, default="3,4,5")
    parser.add_argument('-dropout_r', type=float, default=0.5)
    parser.add_argument('-num_classes', type=int, default=2)

    parser.add_argument('-pi_decay_factor', type=float, default=0.99)
    parser.add_argument('-pi_lower_bound', type=float, default=0.)

    args = parser.parse_args()

    return args
