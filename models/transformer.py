"""Transformer encoder model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert image into patches.
    [B, 1, 28, 28] -> [B, num_patches, patch_size * patch_size]
    Each patch is flattened into a vector of size patch_size * patch_size.
    """
    x = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    """
    This is a transformer encoder that takes an image as input and outputs a class logit.
    Patch embedding + cls token + learnable positions + stacked TransformerEncoder layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        patch_size: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (28 // patch_size) ** 2
        in_dim = patch_size * patch_size
        self.patch_proj = nn.Linear(in_dim, d_model) # project each patch into a d_model-dimensional space
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        patches = patchify(x, self.patch_size)
        tok = self.patch_proj(patches)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, tok), dim=1)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
