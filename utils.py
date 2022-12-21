import os
import json
import random
import argparse
import torch
import torchvision
import numpy as np
from PIL import Image
from torchinfo import summary


def save_model_stats(model: torch.nn.Module, path: str, metadata: dict, device: torch.device,
                     args: argparse.Namespace) -> None:
    try:
        sample_x0 = torch.randn((args.batch_size,) + metadata['data_shape'])
        sample_t = torch.randint(1, args.noise_step, (args.batch_size,))

        model_stats = summary(model.train(),
                              input_data=[(sample_x0, sample_t)],
                              device=device,
                              verbose=0)

        model_stats = str(model_stats)
        with open(path, 'w') as f:
            f.write(model_stats)
    except:
        print("Fail to save model stats. Some modules may not support torchinfo.")


def save_args_to_json(args: argparse.Namespace, path: str) -> None:
    args = vars(args)
    with open(path, 'w') as f:
        json.dump(args, f, indent=4)


def save_grid_image(images: torch.Tensor, path: str, nrow: int = 4, padding: int = 2) -> None:
    grid = torchvision.utils.make_grid(
        images,  # (B, C, H, W)
        nrow=nrow,
        padding=padding,
    )
    grid = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(grid)
    im.save(path)


def set_seed(seed: int, is_cudnn_deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if is_cudnn_deterministic:  # could slow down
        torch.backends.cudnn.deterministic = True


def get_results_path(save_folder: str) -> str:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_results = os.listdir(save_folder)

    if len(all_results) == 0:
        results_path = os.path.join(save_folder, 'results_00')
    else:
        final_results_idx = max([int(result.split('_')[-1]) for result in all_results])
        results_path = os.path.join(save_folder, f'results_{str(final_results_idx + 1).zfill(2)}')

    os.makedirs(results_path)
    os.makedirs(os.path.join(results_path, 'png'))
    os.makedirs(os.path.join(results_path, 'ckpt'))

    return results_path


def get_data(dataset_name: str, **kwargs: dict):  # -> tuple[torch.utils.data.Dataset, dict, callable]:
    if dataset_name == 'cifar10':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        resize = 32 if kwargs['resize'] is None else int(kwargs['resize'])

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=os.path.join('data', 'cifar'),
            train=True,
            download=True,
            transform=transform
        )
        metadata = {
            'data_shape': (3, resize, resize),
            'num_classes': 10,
            'mean': mean,
            'std': std,
        }
        collate_fn = None

        return dataset, metadata, collate_fn

    else:
        raise NotImplementedError
