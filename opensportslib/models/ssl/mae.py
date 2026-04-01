# opensportslib/models/ssl/mae.py

"""Masked Autoencoder (VideoMAE-style) for self-supervised video pre-training.

Implements tube masking and random masking strategies with a ViT encoder
and a lightweight decoder that reconstructs masked patches in pixel space.
"""

import math
import torch
import torch.nn as nn
from functools import partial


class PatchEmbed3D(nn.Module):
    """Convert a video clip into a sequence of patch embeddings.

    Args:
        img_size: spatial resolution (H, W).
        patch_size: spatial patch size.
        num_frames: temporal length of the input clip.
        tubelet_size: temporal size of each 3-D patch (tube).
        in_channels: number of input channels.
        embed_dim: embedding dimension.
    """

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
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

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
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, D, t, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
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


class VideoMAE(nn.Module):
    """VideoMAE: Masked Autoencoder for self-supervised video pre-training.

    The encoder processes only the visible (unmasked) patches, and a
    lightweight decoder reconstructs the masked patches in pixel space.

    Args:
        img_size: spatial resolution (H, W).
        patch_size: spatial patch size.
        num_frames: temporal length of the input clip.
        tubelet_size: temporal size of each 3-D patch.
        in_channels: number of input channels.
        encoder_embed_dim: encoder embedding dimension.
        encoder_depth: number of encoder transformer blocks.
        encoder_num_heads: number of encoder attention heads.
        decoder_embed_dim: decoder embedding dimension.
        decoder_depth: number of decoder transformer blocks.
        decoder_num_heads: number of decoder attention heads.
        mask_ratio: fraction of patches to mask.
        mask_type: masking strategy ("random" or "tube").
        norm_pix_loss: if True, normalize patch pixels before loss.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_channels=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mask_ratio=0.9,
        mask_type="tube",
        norm_pix_loss=True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        # --- encoder ---
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_embed_dim)
        )
        self.encoder_blocks = nn.ModuleList(
            [TransformerBlock(encoder_embed_dim, encoder_num_heads) for _ in range(encoder_depth)]
        )
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # --- decoder ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [TransformerBlock(decoder_embed_dim, decoder_num_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # prediction head: project back to pixel space
        pixels_per_patch = tubelet_size * patch_size * patch_size * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, pixels_per_patch)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # masking
    # ------------------------------------------------------------------

    def _random_masking(self, x, mask_ratio):
        """Per-sample random masking."""
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def _tube_masking(self, x, mask_ratio):
        """Tube masking: mask the same spatial locations across all frames."""
        B, N, D = x.shape
        t, h, w = self.patch_embed.grid_size
        spatial_patches = h * w
        num_keep_spatial = int(spatial_patches * (1 - mask_ratio))

        noise = torch.rand(B, spatial_patches, device=x.device)
        ids_shuffle_spatial = torch.argsort(noise, dim=1)
        ids_restore_spatial = torch.argsort(ids_shuffle_spatial, dim=1)

        ids_keep_spatial = ids_shuffle_spatial[:, :num_keep_spatial]

        # expand across temporal dimension
        ids_keep = []
        for ti in range(t):
            ids_keep.append(ids_keep_spatial + ti * spatial_patches)
        ids_keep = torch.cat(ids_keep, dim=1)  # (B, t * num_keep_spatial)

        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # build full restore indices
        ids_restore_list = []
        for ti in range(t):
            ids_restore_list.append(ids_restore_spatial + ti * spatial_patches)
        ids_restore = torch.cat(ids_restore_list, dim=1)

        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)

        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward_encoder(self, x):
        """Encode only the visible patches.

        Args:
            x: (B, C, T, H, W) input video clip.

        Returns:
            encoded visible tokens, binary mask, restore indices.
        """
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.encoder_pos_embed[:, 1:, :]

        if self.mask_type == "tube":
            x, mask, ids_restore = self._tube_masking(x, self.mask_ratio)
        else:
            x, mask, ids_restore = self._random_masking(x, self.mask_ratio)

        # prepend cls token
        cls = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decode: insert mask tokens, add positional embeddings, reconstruct.

        Args:
            x: encoder output (B, 1 + num_visible, encoder_dim).
            ids_restore: indices to unshuffle tokens back to original order.

        Returns:
            reconstructed patches (B, N, pixels_per_patch).
        """
        x = self.decoder_embed(x)

        # append mask tokens
        B, _, D = x.shape
        N = self.patch_embed.num_patches
        mask_tokens = self.mask_token.expand(B, N + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # prepend cls

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    def patchify(self, video):
        """Convert video to patch targets for reconstruction loss.

        Args:
            video: (B, C, T, H, W).

        Returns:
            patches: (B, N, tubelet_size * patch_size * patch_size * C).
        """
        p = self.patch_size
        t = self.tubelet_size
        c = self.in_channels

        B, C, T, H, W = video.shape
        nt = T // t
        nh = H // p
        nw = W // p

        # (B, C, nt, t, nh, p, nw, p) -> (B, nt*nh*nw, t*p*p*C)
        x = video.reshape(B, c, nt, t, nh, p, nw, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # (B, nt, nh, nw, t, p, p, C)
        x = x.reshape(B, nt * nh * nw, t * p * p * c)
        return x

    def forward(self, x):
        """Full forward pass: encode visible patches, decode all patches.

        Args:
            x: (B, C, T, H, W) input video clip.

        Returns:
            dict with keys:
                - "pred": reconstructed patches (B, N, pixels_per_patch).
                - "target": ground truth patches (B, N, pixels_per_patch).
                - "mask": binary mask (B, N), 1 = masked.
        """
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)

        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        return {"pred": pred, "target": target, "mask": mask}

    def get_encoder(self):
        """Return a standalone encoder (without decoder) for downstream use."""
        return _MAEEncoder(self)


class _MAEEncoder(nn.Module):
    """Wrapper that exposes only the MAE encoder for downstream fine-tuning."""

    def __init__(self, mae_model):
        super().__init__()
        self.patch_embed = mae_model.patch_embed
        self.cls_token = mae_model.cls_token
        self.pos_embed = mae_model.encoder_pos_embed
        self.blocks = mae_model.encoder_blocks
        self.norm = mae_model.encoder_norm

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # cls token output
