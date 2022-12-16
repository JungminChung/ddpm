import os 
import argparse

import torch 
from torch.utils.data import DataLoader

from unet import UNet
from ddpm import Diffusion
from utils import set_seed, get_results_path, get_data

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_folder', type=str, default='save', help='save folder path for trained results')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='train dataset name')
    parser.add_argument('--img_resize_size', default=None, help='if given resize image size when target dataset is kind of img dataset, else no resize')
    
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--epoch', type=int, default=300, help='train epoch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--noise_step', type=int, default=1000, help='total number of noise steps')
    return parser.parse_args()

def main(): 
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)
    results_path = get_results_path(args.save_folder)

    dataset, metadata, collate_fn = get_data(args.dataset, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    diffusion = Diffusion(model, args.noise_step, metadata['data_shape'][1], device)


if __name__=='__main__':
    main()