from maml import train
from utils import setlogger

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='MAML Implementation on CWRU Bearing Dataset')

    parser.add_argument("--log_file", type=str, default="./logs/maml.log", help="log file path")

    parser.add_argument('--ways', type=int, default=10, help='Number of classes in a task')
    parser.add_argument('--shots', type=int, default=5, help='Number of samples in a class')
    parser.add_argument('--fast_lr', type=float, default=0.01, help='Learning rate for the inner loop')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate for the outer loop')
    parser.add_argument('--adapt_steps', type=int, default=5, help='Number of inner loop steps')
    parser.add_argument('--meta_batch_size', type=int, default=32, help='Number of tasks in a batch')
    parser.add_argument('--iters', type=int, default=4000, help='Number of outer loop iterations')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger(args.log_file)

    train(
        ways=args.ways,
        shots=args.shots,
        meta_lr=args.meta_lr,
        fast_lr=args.fast_lr,
        adapt_steps=args.adapt_steps,
        meta_batch_size=args.meta_batch_size,
        iters=args.iters,
        cuda=args.cuda,
        seed=args.seed
    )