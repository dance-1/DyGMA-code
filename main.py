import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from solver import Solver
from utils.utils import mkdir


def fix_random_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    cudnn.benchmark = True

    if config.mode == 'train':
        mkdir(config.checkpoint_dir)
    mkdir(config.output_dir)

    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        score_file = solver.test()
        from evaluate import evaluate_file

        print("======================EVALUATION MODE======================", flush=True)
        print(f"Evaluating score file: {score_file}", flush=True)
        evaluate_file(
            score_file=score_file,
            base_dir=os.path.dirname(score_file),
            win_size=config.win_size,
        )

    return solver


def build_parser():
    parser = argparse.ArgumentParser(description='Train or evaluate DyGMA for multivariate time-series anomaly detection.')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--patch_len', type=int, default=10)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='dataset/MSL')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=os.path.join('checkpoints', 'pretrained'),
        help='Directory containing pretrained checkpoints or receiving trained checkpoints.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory for NPY score artifacts and text reports.',
    )

    return parser


if __name__ == '__main__':
    fix_random_seed()

    parser = build_parser()
    config = parser.parse_args()

    print('------------ Options -------------')
    for name, value in sorted(vars(config).items()):
        print(f'{name}: {value}')
    print('-------------- End ----------------')

    main(config)
