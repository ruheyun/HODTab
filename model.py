import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from diffusion import GaussianDiffusion
from backbones import DiffSynthesizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torch.optim as optim


class TabDiff:
    def __init__(
            self,
            d_in,
            classes=None,
            timestep_emb_dim: int = 64,
            diffusion_total_steps: int = 1000,
            diffusion_scheduler: str = 'linear',
            device: str = 'cpu',
            num_epochs: int = 100,
            learning_rate: float = 1e-4,
            batch_size: int = 128,
            batch_size_sample: int | None = None,
            backbone_type: str = 'mlp',
            backbone_config: dict | None = None,
        ):
        self.learning_rate = learning_rate
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_sample = batch_size_sample  # default sample batch size same as training batch size

        # init the synthesizer model
        self.synthesizer = DiffSynthesizer(
            d_in=d_in,
            n_classes=classes,
            time_embed_dim=timestep_emb_dim,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
        )

        # push model to compute device
        self.synthesizer = self.synthesizer.to(self.device)

        # init the diffuser
        self.diffuser = GaussianDiffusion(
            input_dim=d_in,
            denoise_fn=self.synthesizer,
            num_timesteps=diffusion_total_steps,
            scheduler=diffusion_scheduler,
            device=device,
        )

        # init the adam model optimizer
        self.optimizer = optim.AdamW(self.diffuser.parameters(), lr=self.learning_rate)

        # init the cosine learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        # placeholder for training history
        self.epoch_loss_history = pd.DataFrame(columns=['epoch', 'loss'])
        self.step_loss_history = pd.DataFrame(columns=['step', 'loss'])


    def fit(self, train_dataloader):
        """ Fit function for training the Diff model

        Args:
            train_dataloader (DataLoader): PyTorch DataLoader containing training data
        """

        # init the training progress bar
        pbar = tqdm(iterable=range(self.num_epochs), position=0, leave=True)

        # iterate over training epochs
        for epoch in pbar:

            loss = self.train_epoch(
                dataloader=train_dataloader,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
            )
                
            # determine mean training round loss
            pbar.set_description(f"Epoch {epoch + 1:04d} | Train Loss: {loss:.6f}")

            # save training history
            self.epoch_loss_history.loc[len(self.epoch_loss_history)] = {'epoch': epoch + 1, 'loss': loss}

    def train_epoch(
            self,
            dataloader,
            optimizer,
            scheduler,
        ):
        """Training module for single epoch, update model parameters and return losses

        Args:
            dataloader (DataLoader): torch Dataloader
            optimizer (Optimizer): optimizer
            scheduler (LRScheduler): learning rate scheduler

        Returns:
            float: mean loss for the epoch
        """
        total_losses = []

        # iterate over distinct mini-batches
        for (x,) in dataloader:
            # set network in training mode
            self.synthesizer.train()
            # push batch to compute device
            x = x.to(self.device)
            y=None
            if y is not None:
                y = y.long().to(self.device)
            else:                
                y = None

            # reset encoder and decoder gradients
            optimizer.zero_grad(set_to_none=True)

            # run error back-propagation
            loss = self.diffuser.compute_loss(x, y)
            loss.backward()

            # optimize encoder and decoder parameters
            optimizer.step()

            # collect rec error losses
            total_losses.append(loss.detach().cpu().numpy())

            self.step_loss_history.loc[len(self.step_loss_history)] = {'step': len(self.step_loss_history) + 1, 'loss': loss.detach().cpu().numpy()}

        # average of rec errors
        total_losses_mean = np.mean(np.array(total_losses))

        # update learning rate according to the scheduler
        scheduler.step()

        return total_losses_mean

    @torch.no_grad()
    def sample(self, n_samples:int | None=None, label: torch.Tensor | None=None):
        """ Generation of samples. 
            For unconditional sampling use n_samples, for conditional sampling provide label.

        Args:
            n_samples (int, optional): number of samples to sample. Defaults to None.
            label (tensor, optional): list of labels for conditional sampling. Defaults to None.

        Raises:
            Exception: If neither n_samples nor label is provided.

        Returns:
            pd.DataFrame: decoded sample dataframe
        """

        if (n_samples is None) and (label is None):
            raise Exception("either n_samples or label needs to be given")

        if label is not None:
            n_samples = len(label)
            label = label.to(device=self.device)

        x_gen, y_gen = self.diffuser.sample_all(n_samples, self.batch_size_sample)

        return x_gen, y_gen
    