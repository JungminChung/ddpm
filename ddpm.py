import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Optional


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, noise_steps: int, img_size: int, device: torch.device,
                 beta_start: Optional[float] = 0.0001, beta_end: Optional[float] = 0.02):
        super().__init__()
        self.model = model
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.beta = self.beta.view(self.beta.shape[0], 1, 1, 1)
        self.sigma_sq = self.beta

        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_x_t(self, x0: torch.Tensor, t: torch.Tensor):
        '''
        Input : 
            x0: [N, C, H, W]
            t: [N]
        Output : 
            x_t: [N, C, H, W]
            eps: [N, C, H, W]
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_bar[t])
        eps = torch.randn_like(x0)
        x_t = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * eps
        return x_t, eps

    def sample_timesteps(self, batch_size: int):
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)

    def sampling(self, model: nn.Module, num_imgs: int):
        model.eval()
        with torch.no_grad():
            x_t = torch.randn(num_imgs, 3, self.img_size, self.img_size, device=self.device)

            for timestep in tqdm(range(self.noise_steps - 1, 0, -1), desc='Sampling'):
                z = torch.randn_like(x_t) if timestep > 1 else torch.zeros_like(x_t)

                alpha = self.alpha[timestep]
                alpha_bar = self.alpha_bar[timestep]
                t = torch.full((num_imgs,), timestep, device=self.device, dtype=torch.long)
                eps_theta = model(x_t, t)  # predict noise at timestep t
                sigma_sq = self.sigma_sq[timestep]

                x_t = 1 / torch.sqrt(alpha) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps_theta) + torch.sqrt(sigma_sq) * z

        model.train()

        x_0 = (x_t.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
        x_0 = (x_0 * 255).to(torch.uint8).cpu()  # [0, 1] -> [0, 255]

        return x_0  # [N, C, H, W]
