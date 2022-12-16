import os 
import torch 
import torchvision
import numpy as np 
import random 

def save_grid_image(images: torch.Tensor, path: str, nrow: int = 4, padding: int = 2) -> None: 
    grid = torchvision.utils.make_grid(
            images, # (B, C, H, W)
            nrow=4, 
            padding=2, 
        )
    torchvision.utils.save_image(grid, path)

def set_seed(seed: int, is_cudnn_deterministic: bool =False) -> None: 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if is_cudnn_deterministic : # could slow-down
        torch.backends.cudnn.deterministic = True

def get_results_path(save_folder: str) -> str: 
    if not os.pah.exists(save_folder):
        os.makedirs(save_folder)

    all_results = os.listdir(save_folder)

    if len(all_results) == 0: 
        results_path = os.path.join(save_folder, 'results_0')
        os.makedirs(results_path)
        return results_path

    final_results_idx = max([int(result.split('_')[-1]) for result in all_results])
    results_path = os.path.join(save_folder, f'results_{final_results_idx}')
    os.makedirs(results_path)
    return results_path
    