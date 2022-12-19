import os
import torch
import torchvision
import numpy as np
import random
from PIL import Image


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
        results_path = os.path.join(save_folder, 'results_0')
        os.makedirs(results_path)
        return results_path

    final_results_idx = max([int(result.split('_')[-1]) for result in all_results])
    results_path = os.path.join(save_folder, f'results_{final_results_idx + 1}')
    os.makedirs(results_path)
    return results_path


def get_data(dataset_name: str, **kwargs: dict):  # -> tuple[torch.utils.data.Dataset, dict, callable]:
    if dataset_name == 'cifar10':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        resize_size = 32 if kwargs['img_resize_size'] is None else kwargs['img_resize_size']

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_size),
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
            'data_shape': (3, resize_size, resize_size),
            'num_classes': 10,
            'mean': mean,
            'std': std,
        }
        collate_fn = None

        return dataset, metadata, collate_fn

    else:
        raise NotImplementedError
