import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from unet import UNet
from ddpm import DDPM
from utils import set_seed, get_results_path, get_data, save_grid_image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_folder', type=str, default='save', help='save folder path for trained results')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='train dataset name')
    parser.add_argument('--img_resize_size', default=None,
                        help='if given resize image size when target dataset is kind of img dataset, else no resize')

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

    dataset, metadata, collate_fn = get_data(args.dataset, **vars(args))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    diffusion = DDPM(model, args.noise_step, metadata['data_shape'][1], device)

    for epoch in range(args.epoch):
        progress = tqdm(dataloader)
        for i, data in enumerate(progress):
            x_0 = data[0].to(device)
            timestep = diffusion.sample_timesteps(x_0.shape[0]).to(device)
            x_t, noise = diffusion.get_x_t(x_0, timestep)
            predicted_noise = model(x_t, timestep)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_description(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

        sampled_data = diffusion.sampling(model, 16)
        save_grid_image(sampled_data, os.path.join(results_path, f'{epoch}.png'))
        torch.save(model.state_dict(), os.path.join(results_path, f'{epoch}.pth'))


if __name__ == '__main__':
    main()
