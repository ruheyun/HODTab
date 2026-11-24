import math
import numpy as np
from tqdm import tqdm
from utils import *

eps = 1e-8


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    
    if schedule_name == "linear":
        
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    
    elif schedule_name == "cosine":

        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    
    else:

        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    
    betas = []

    for i in range(num_diffusion_timesteps):

        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return np.array(betas)


class GaussianDiffusion(torch.nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        denoise_fn,
        num_timesteps=1000,
        scheduler='cosine',
        device=torch.device('cpu'),
    ):
        
        super().__init__()

        self.input_dim = input_dim

        self._denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler

        betas = get_named_beta_schedule(scheduler, num_timesteps)

        betas = torch.tensor(betas.astype('float64'))
        alphas = 1. - betas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        
        self.posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        ).float().to(device)
        
        self.posterior_mean_coef1 = (
                betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        
        self.posterior_mean_coef2 = (
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas.numpy()) / (1.0 - alphas_cumprod)
        ).float().to(device)
        
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5
        
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))
        
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
        
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def gaussian_p_mean_variance(self, model_output, x, t):

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x.device), (1. - self.alphas)[1:]],
                                dim=0)
        
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
        
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
        
    
    def _gaussian_loss(self, model_out, noise):
        loss = mean_flat((noise - model_out) ** 2)
        return loss
    
    
    def gaussian_p_sample(self, model_out, x, t):
        out = self.gaussian_p_mean_variance(model_out, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"]
        }
        
    def sample_time(self, b, device):
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps

        return t, pt
    
    def compute_loss(self, x, out_dict):
        b = x.shape[0]
        device = x.device

        t, pt = self.sample_time(b, device)

        noise = torch.randn_like(x)
        x_t = self.gaussian_q_sample(x, t, noise=noise)

        model_out = self._denoise_fn(
            x_t,
            t,
            out_dict
        )

        losses = self._gaussian_loss(model_out, noise)

        return losses.mean()

    def sample_all(self, num_samples, batch_size, y_dist=None, y_conditions=None, max_attempts=10):
        if exists(y_dist) and exists(y_conditions):
            raise ValueError('y_dist and y_conditions cannot be provided together')    

        if exists(y_conditions):
            y_conditions = torch.as_tensor(y_conditions, dtype=torch.long, device='cpu').reshape(-1)

            if len(y_conditions) != num_samples:
                raise ValueError('Number of conditions must match num_samples')
        
        x_gen = torch.empty(
            (num_samples, self.input_dim),
            dtype=torch.float32,
        )

        has_labels = (
            exists(y_dist) or exists(y_conditions)
        )

        y_gen = (
            torch.empty(num_samples, dtype=torch.long)
            if has_labels else None
        )

        with tqdm(total=num_samples, desc='Sampling', unit='sample') as pbar:
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)

                pending = torch.arange(end - start)

                for _ in range(max_attempts):
                    if pending.numel() == 0:
                        break

                    batch_y = (
                        y_conditions[start:end][pending]
                        if exists(y_conditions) else None
                    )

                    generated, out_dict = self.sample(
                        num_samples=pending.numel(),
                        y_dist=y_dist,
                        y_conditions=batch_y,
                    )

                    generated = generated.detach().cpu()
                    valid = torch.isfinite(generated).all(dim=1)

                    positions = start + pending[valid]
                    x_gen[positions] = generated[valid].to(x_gen.dtype)

                    if exists(y_gen):
                        labels = out_dict['y'].detach().cpu()
                        y_gen[positions] = labels[valid]

                    pbar.update(int(valid.sum().item()))
                    pending = pending[~valid]

                if pending.numel() > 0:
                    raise RuntimeError('Failed to generate valid samples')
        return x_gen, y_gen
    
    
    @torch.no_grad()
    def sample(self, num_samples, y_dist=None, y_conditions=None):
        b = num_samples
        device = self.log_alpha.device

        if y_dist is not None and y_conditions is not None:
            raise ValueError('y_dist and y_conditions cannot be provided together')
        
        z_norm = torch.randn((b, self.input_dim), device=device)

        out_dict = {}

        if exists(y_conditions):
            y = torch.as_tensor(y_conditions, dtype=torch.long, device=device).reshape(-1)

            if len(y) != b:
                raise ValueError('Number of conditions must match num_samples')

            out_dict = {'y': y}

        elif exists(y_dist):
            y = torch.multinomial(y_dist, num_samples=b, replacement=True)
            out_dict = {'y': y.long().to(device)}

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                z_norm.float(),
                t,
                **out_dict
            )
            z_norm = self.gaussian_p_sample(model_out, z_norm, t)['sample']

        return z_norm.cpu(), out_dict
        