# opensportslib/models/ssl/contrastive.py

"""SimCLR-style contrastive learning for video pre-training.

Two augmented views of the same clip are passed through a shared ViT
encoder and a projection head. The NT-Xent (normalized temperature-scaled
cross-entropy) loss pushes representations of the same clip together and
apart from other clips in the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """Convert a video clip into a sequence of patch embeddings."""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_channels=3,
        embed_dim=768,
    ):
        super().__init__()
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SimCLRProjectionHead(nn.Module):
    """Two-layer MLP projection head for SimCLR."""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """SimCLR contrastive learning model for video pre-training.

    Args:
        img_size: spatial resolution (H, W).
        patch_size: spatial patch size.
        num_frames: temporal length of the input clip.
        tubelet_size: temporal size of each 3-D patch.
        in_channels: number of input channels.
        embed_dim: ViT embedding dimension.
        depth: number of transformer blocks.
        num_heads: number of attention heads.
        proj_hidden_dim: projection head hidden dimension.
        proj_out_dim: projection head output dimension.
        temperature: NT-Xent temperature.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        proj_hidden_dim=2048,
        proj_out_dim=128,
        temperature=0.1,
    ):
        super().__init__()
        self.temperature = temperature

        # encoder
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # projection head
        self.projector = SimCLRProjectionHead(embed_dim, proj_hidden_dim, proj_out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        """Encode a video clip to a CLS-token representation.

        Args:
            x: (B, C, T, H, W) video clip.

        Returns:
            (B, embed_dim) representation.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def nt_xent_loss(self, z1, z2):
        """Compute the NT-Xent contrastive loss.

        Args:
            z1: (B, D) projections from view 1.
            z2: (B, D) projections from view 2.

        Returns:
            scalar loss.
        """
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # mask out self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss

    def forward(self, views):
        """Forward pass for SimCLR.

        Args:
            views: list of 2 augmented views, each (B, C, T, H, W).

        Returns:
            dict with keys:
                - "loss": NT-Xent loss (scalar).
                - "z1": projections from view 1.
                - "z2": projections from view 2.
        """
        h1 = self.encode(views[0])
        h2 = self.encode(views[1])

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss = self.nt_xent_loss(z1, z2)

        return {"loss": loss, "z1": z1, "z2": z2}

    def get_encoder(self):
        """Return a standalone encoder for downstream use."""
        return _SimCLREncoder(self)


class _SimCLREncoder(nn.Module):
    """Wrapper that exposes only the SimCLR encoder for downstream fine-tuning."""

    def __init__(self, simclr_model):
        super().__init__()
        self.patch_embed = simclr_model.patch_embed
        self.cls_token = simclr_model.cls_token
        self.pos_embed = simclr_model.pos_embed
        self.blocks = simclr_model.blocks
        self.norm = simclr_model.norm

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
