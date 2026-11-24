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
            data_transformer,
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
            train_dataloader: torch.utils.data.DataLoader | None = None
        ):
        self.learning_rate = learning_rate
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_sample = batch_size_sample  # default sample batch size same as training batch size
        self.data_transformer = data_transformer
        self.train_dataloader = train_dataloader

        # init the synthesizer model
        self.synthesizer = DiffSynthesizer(
            num_cols_dim=len(data_transformer.numerical_cols),
            n_classes=data_transformer.label_cardinality_,
            time_embed_dim=timestep_emb_dim,
            backbone_type=backbone_type,
            backbone_config=backbone_config,
        )

        # push model to compute device
        self.synthesizer = self.synthesizer.to(self.device)

        # init the diffuser
        self.diffuser = GaussianDiffusion(
            input_dim=len(data_transformer.numerical_cols),
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
        self.loss_history = pd.DataFrame(columns=['epoch', 'step', 'loss'])


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
            self.training_history["train_loss"].append(loss)

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
        for batch in dataloader:
            # set network in training mode
            self.synthesizer.train()
            # push batch to compute device
            batch_cat = batch["cat"].to(self.device)
            batch_num = batch["num"].to(self.device)
            if "label" in batch:
                batch_y = batch["label"].to(self.device)
            else:                
                batch_y = None

            # reset encoder and decoder gradients
            optimizer.zero_grad()

            # run error back-propagation
            loss = self.diffusion.compute_loss(x, out_dict)

            # optimize encoder and decoder parameters
            optimizer.step()

            # collect rec error losses
            total_losses.append(train_losses.detach().cpu().numpy())

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

        # logits for cat output when one-hot output is used        
        cat_logits = None
        # get dimension of encoded data
        # dim_input = self.synthesizer.dim_input
        dim_input = len(self.data_transformer.numerical_cols) + self.cat_emb_dim * self.cat_cols_dim

        # ensure n_samples is available for sample generation
        assert n_samples is not None
        chunk_size: int = self.batch_size_sample if self.batch_size_sample is not None else n_samples
        assert chunk_size is not None

        # decide chunk size for batch-wise processing
        total_chunks = int(np.ceil(n_samples / chunk_size))

        decoded_list = []

        # Process samples in chunks to reduce peak memory usage.
        for chunk_idx, start in enumerate(range(0, n_samples, chunk_size)):
            cur_n = min(chunk_size, n_samples - start)

            # slice labels for current chunk if provided
            label_chunk = None
            if label is not None:
                label_chunk = label[start: start + cur_n]

            # init random noise for this chunk
            z = torch.randn((cur_n, dim_input), device=self.device)

            # run the reverse diffusion for this chunk
            pbar = tqdm(iterable=reversed(range(0, self.diffuser.total_steps)),
                        desc=f"SAMPLING chunk {start}-{start+cur_n-1}", leave=False)
            for i in pbar:
                pbar.set_description(f"SAMPLING STEP: {i:4d} (chunk ({chunk_idx+1}/{total_chunks}) {start}-{start+cur_n-1})")

                t = torch.full((cur_n,), i, dtype=torch.long, device=self.device)

                if self.cat_decoding == 'distance':
                    model_out = self.synthesizer(z.float(), t, label_chunk)
                elif self.cat_decoding == 'logits':
                    model_out, cat_logits = self.synthesizer(z.float(), t, label_chunk)
                else:
                    raise ValueError(f"Unknown cat_decoding: {self.cat_decoding}")

                z = self.diffuser.p_sample_gauss(model_out, z, t)

            # decode samples for the chunk and collect
            df_chunk = self.decode_sample(sample=z, cat_logits=cat_logits)
            decoded_list.append(df_chunk)

        # concatenate decoded chunks preserving order
        if len(decoded_list) > 0:
            x_0 = pd.concat(decoded_list, ignore_index=True)
        else:
            x_0 = pd.DataFrame()

        return x_0
    
    def decode_sample(
            self,
            sample,
            cat_logits=None,
        ):
        """ Decoding function for unscaling numeric attributes and inverse encoding of categorical attributes.
            Used once synthetic data is generated. 

        Args:
            sample (tensor): input samples for decoding
            cat_logits (tensor, optional): predicted categorical logits for decoding when one-hot encoding is used. Defaults to None.

        Returns:
            pandas DataFrame: decoded dataframe
        """
        cat_dim = self.cat_emb_dim * self.cat_cols_dim
        # split sample into numeric and categorical parts
        sample = sample.cpu().numpy()
        sample_num = sample[:, cat_dim:]
        sample_cat = sample[:, :cat_dim]

        # decode categorical embeddings to categorical codes
        if self.cat_decoding == 'distance':
            sample_cat = self.decode_cat_emb_distance(sample_cat)
        elif self.cat_decoding == 'logits':
            sample_cat = self.decode_cat_emb_logits(cat_logits)
        else:
            raise ValueError(f"Categorical decoding {self.cat_decoding} not supported.")
        
        # inverse transform to original data space
        sample_decoded = self.data_transformer.inverse_transform({"cat": sample_cat, "num": sample_num})

        return sample_decoded

    def decode_cat_emb_distance(
            self,
            sample_cat
        ):
        """ Decoding function for inverse encoding of categorical attributes from embeddings.
            Used once synthetic data is generated. 

        Args:
            sample_cat (np.ndarray): input categorical samples for decoding

        Returns:
            np.ndarray: decoded categorical data array
        """
        # Check if the model is wrapped by Opacus or similar module wrappers
        if hasattr(self.synthesizer, "_module"):
            # get embedding lookup matrix
            embedding_lookup = self.synthesizer._module.get_x_cat_emb().cpu()
        else:
            embedding_lookup = self.synthesizer.get_x_cat_emb().cpu()

        # embedding_lookup = self.synthesizer.get_x_cat_emb().cpu()
        # categorical attributes
        cat_attrs = self.data_transformer.embedding_mapping_.keys()
        # reshape back to batch_size * n_dim_cat * cat_emb_dim
        sample_cat = sample_cat.reshape(-1, len(cat_attrs), self.cat_emb_dim)
        # compute batch-wise calculation of distances for memory efficiency
        batch_size = self.batch_size
        n_samples = len(sample_cat)
        z_cat_list = []

        # iterate over generated categorical samples
        for i in range(0, n_samples, batch_size):
            # get batch of samples
            samples_cat_subset = sample_cat[i: i+batch_size]
            # get the closest distance based on the embeddings that belong to a column category
            distances = torch.cdist(x1=embedding_lookup, x2=torch.as_tensor(samples_cat_subset, dtype=torch.float32))
            z_cat = []
            for attr_idx, attr_name in enumerate(cat_attrs):
                # get vocab indices for attribute
                attr_emb_idx = list(self.data_transformer.embedding_mapping_[attr_name])
                # get distances for attribute
                attr_distances = distances[:, attr_emb_idx, attr_idx]
                # get nearest embedding index
                _, nearest_idx = torch.min(attr_distances, dim=1)
                # convert to numpy
                nearest_idx = nearest_idx.cpu().numpy()
                # map emb indices back to column indices
                z_cat.append(np.array(attr_emb_idx)[nearest_idx])

            z_cat = np.array(z_cat).T
            z_cat_list.append(z_cat)
        z_cat = np.vstack(z_cat_list)

        return z_cat

    def decode_cat_emb_logits(
            self,
            cat_logits
        ):
        """
        Decoding function for inverse encoding of categorical attributes from logits.

        Args:
            cat_logits (list of torch.Tensor): List of predicted categorical logits.

        Returns:
            np.ndarray: Decoded categorical codes.
        """
        z_cat = []

        for cat_idx, cat_logit in enumerate(cat_logits):
            cat_logit = torch.argmax(cat_logit, dim=1)
            # add min index of cat column
            cat_min_idx = list(self.data_transformer.embedding_mapping_.values())[cat_idx].min()
            cat_logit += cat_min_idx
            z_cat.append(cat_logit)

        z_cat = torch.stack(z_cat, dim=1)
        z_cat = z_cat.cpu().numpy()

        return z_cat
    