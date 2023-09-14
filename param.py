import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--path', type=str, default='./HMDD3')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--MLPDropout', type=float, default=0.5)
    parser.add_argument('--status', type=int, default=1)
    return parser.parse_args()