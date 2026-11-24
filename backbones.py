import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal time embedding module.
    
    Args:
        dim (int): The dimension of the sinusoidal embeddings. Must be even.
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Sinusoidal dimension must be even"
        self.dim = dim

    def forward(self, time):
        """
        Forward pass for sinusoidal position embeddings.
        
        Args:
            time (torch.Tensor): Timesteps of shape [batch_size].
            
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape [batch_size, dim].
        """
        # time shape: [batch_size]
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # embeddings shape: [batch_size, dim]
        return embeddings


class ResidualBlock(nn.Module):
    """A simple residual block for the MLP."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.activation(self.layer1(x))
        out = self.layer2(out)
        return identity + out
    

class MLPBackbone(nn.Module):
    """
    MLP Backbone.
    
    Args:
        num_features (int): The dimension of the input tabular features.
        out_features (int): The dimension of the output tabular features.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
        num_blocks (int, optional): Number of residual blocks. Defaults to 4.
    """
    def __init__(self, num_features, out_features, hidden_dim=256, num_blocks=4):
        super().__init__()
        
        # Input projection now takes features + condition embedding
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        # x shape: [batch_size, num_features]
        
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h)
            
        output = self.output_proj(h)

        return output


class TransformerBackbone(nn.Module):
    """
    Transformer Backbone.
    
    Args:
        num_features (int): The dimension of the input tabular features.
        d_model (int, optional): The dimension of the transformer model. Defaults to 256.
        n_head (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of transformer encoder layers. Defaults to 6.
        dim_feedforward (int, optional): Dimension of the feedforward network model. Defaults to 1024.
    """
    def __init__(self, num_features, cond_dim, d_model=256, n_head=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        # Project conditioning embedding to d_model
        self.cond_proj = nn.Linear(cond_dim, d_model)
        
        self.feature_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(num_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, cond_emb):
        # x shape: [batch_size, num_features]
        # cond_emb shape: [batch_size, cond_dim]
        
        # 1. Project condition to d_model
        cond_emb_proj = self.cond_proj(cond_emb) # [batch_size, d_model]
        
        # 2. Reshape and embed features
        x = x.unsqueeze(-1)
        x_emb = self.feature_embed(x) # [batch_size, num_features, d_model]
        
        # 3. Add positional embedding
        pos_emb = self.pos_embed.weight.unsqueeze(0)
        x_emb = x_emb + pos_emb
        
        # 4. Add condition embedding (broadcasted)
        x_with_cond = x_emb + cond_emb_proj.unsqueeze(1)
        
        # 5. Pass through transformer
        transformer_out = self.transformer_encoder(x_with_cond)
        
        # 6. Project back to noise
        output = self.output_proj(transformer_out)
        
        return output.squeeze(-1)


class UNetBackbone(nn.Module):
    """
    U-Net backbone.
    - Conditioning added as per-layer linear projection and summed pre-activation.
    
    Args:
        num_features (int): The dimension of the input tabular features.
        cond_dim (int): The dimension of the conditioning embedding.
        hidden_dims (list of int, optional): Dimensions of the hidden layers. Defaults to [256, 512].
        activation (callable, optional): Activation function to use. Defaults to nn.SiLU.
    """
    def __init__(self, num_features, cond_dim, hidden_dims=[256, 512], activation=nn.SiLU):
        super().__init__()
        self.num_features = num_features
        self.hidden_dims = list(hidden_dims)

        # --- Encoder (Down Path) ---
        self.down_blocks = nn.ModuleList()
        self.cond_projs = nn.ModuleList()

        # in/out dims for down path: [num_features -> h0, h0 -> h1, ...]
        in_dim = num_features
        for out_dim in self.hidden_dims:
            self.down_blocks.append(nn.Linear(in_dim, out_dim))
            self.cond_projs.append(nn.Linear(cond_dim, out_dim))
            in_dim = out_dim

        # --- Bottleneck ---
        bot_dim = self.hidden_dims[-1] if self.hidden_dims else num_features
        self.bottleneck = nn.Linear(bot_dim, bot_dim)
        self.cond_projs.append(nn.Linear(cond_dim, bot_dim))

        # --- Decoder (Up Path) ---
        # Build sequentially so in_dim matches the runtime concatenation (h + skip)
        self.up_blocks = nn.ModuleList()
        h_dim = bot_dim
        for i in reversed(range(len(self.hidden_dims))):
            skip_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i-1] if i > 0 else num_features
            in_dim = h_dim + skip_dim
            self.up_blocks.append(nn.Linear(in_dim, out_dim))
            self.cond_projs.append(nn.Linear(cond_dim, out_dim))
            h_dim = out_dim  # will be input 'h' width for the next up block

        self.activation = activation()

    def forward(self, x, cond_emb):
        """
        Forward pass for the U-Net backbone.
        
        Args:
            x (torch.Tensor): Input noisy data of shape [batch_size, num_features].
            cond_emb (torch.Tensor): Conditioning embedding of shape [batch_size, cond_dim].
            
        Returns:
            torch.Tensor: Denoised data (or noise prediction) of shape [batch_size, num_features].
        """
        skips = []
        h = x
        cond_iter = iter(self.cond_projs)

        # --- Down Path ---
        for block in self.down_blocks:
            c = next(cond_iter)(cond_emb)      # [B, out_dim]
            h = self.activation(block(h) + c)  # [B, out_dim]
            skips.append(h)                    # store activated output

        # --- Bottleneck ---
        c = next(cond_iter)(cond_emb)          # [B, bot_dim]
        h = self.activation(self.bottleneck(h) + c)

        # --- Up Path ---
        for block in self.up_blocks:
            c = next(cond_iter)(cond_emb)      # [B, out_dim]
            skip = skips.pop()                 # matches decoder depth
            h = torch.cat([h, skip], dim=1)    # [B, in_dim]
            h = self.activation(block(h) + c)  # [B, out_dim]

        return h


class DiffSynthesizer(nn.Module):
    """
    A conditional diffusion model for tabular data with swappable backbones.
    """
    def __init__(
            self, 
            d_in: int, 
            n_classes: int | None = None, 
            time_embed_dim: int | None = None,
            backbone_type: str = 'mlp',            
            backbone_config: dict | None = None,
        ):
        super().__init__()

        if backbone_config is None:
            backbone_config = {}

        if time_embed_dim is None:
            time_embed_dim = 64

        self.proj = nn.Linear(d_in, time_embed_dim)

        # --- 1. Conditioning Embeddings ---
        self.time_embed_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )

        if n_classes is not None:
            self.label_embed = nn.Embedding(n_classes, time_embed_dim)
                
        # --- 2. Backbone Switch ---        
        if backbone_type == 'mlp':
            self.backbone = MLPBackbone(
                num_features=time_embed_dim,
                out_features=d_in,
                hidden_dim=backbone_config.get('mlp_hidden_dim', 512),
                num_blocks=backbone_config.get('mlp_num_residual_blocks', 2)
            )
        elif backbone_type == 'transformer':
            self.backbone = TransformerBackbone(
                num_features=self.dim_input,
                cond_dim=time_embed_dim,
                d_model=backbone_config.get('d_model', 1024),
                n_head=backbone_config.get('transformer_n_head', 2),
                num_layers=backbone_config.get('transformer_num_layers', 2),
                dim_feedforward=backbone_config.get('transformer_dim_feedforward', 4096)
            )
        elif backbone_type == 'unet':
            self.backbone = UNetBackbone(
                num_features=self.dim_input,
                cond_dim=time_embed_dim,
                hidden_dims=backbone_config.get('unet_hidden_dims', [256, 512])
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")         
    
    def forward(self, x, timesteps, label=None):
        # --- 1. Compute Conditioning Embedding ---
        # [batch_size, cond_embed_dim]
        t_emb = self.time_embed_mlp(timesteps)
        
        # [batch_size, cond_embed_dim]
        if label is not None:
            l_emb = self.label_embed(label)
            # Combine embeddings
            cond_emb = t_emb + l_emb
        else:
            cond_emb = t_emb 

        # --- 2. Pass to Backbone ---
        # The backbone predicts the noise
        x = self.proj(x) + cond_emb

        return self.backbone(x)
