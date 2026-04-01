# opensportslib/models/ssl/dino.py

"""DINO: self-distillation with no labels for video pre-training.

A student network is trained to match the output of an exponential
moving average (EMA) teacher. Both see different augmented views of the
same clip. The teacher output is centered and sharpened; the student is
trained with a cross-entropy loss against the teacher's soft targets.
"""

import copy
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


class DINOHead(nn.Module):
    """MLP projection head used by both student and teacher."""

    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class ViTBackbone(nn.Module):
    """ViT encoder backbone shared by student and teacher."""

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
    ):
        super().__init__()
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

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # cls token


class DINO(nn.Module):
    """DINO self-distillation model for video pre-training.

    Args:
        img_size: spatial resolution (H, W).
        patch_size: spatial patch size.
        num_frames: temporal length of the input clip.
        tubelet_size: temporal size of each 3-D patch.
        in_channels: number of input channels.
        embed_dim: ViT embedding dimension.
        depth: number of transformer blocks.
        num_heads: number of attention heads.
        head_hidden_dim: DINO projection head hidden dim.
        head_bottleneck_dim: DINO projection head bottleneck dim.
        head_out_dim: DINO projection head output dim.
        momentum: EMA momentum for teacher update.
        teacher_temp: temperature for teacher softmax.
        student_temp: temperature for student softmax.
        center_momentum: momentum for teacher output centering.
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
        head_hidden_dim=2048,
        head_bottleneck_dim=256,
        head_out_dim=65536,
        momentum=0.996,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.momentum = momentum
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # student
        self.student_backbone = ViTBackbone(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.student_head = DINOHead(
            embed_dim, head_hidden_dim, head_bottleneck_dim, head_out_dim
        )

        # teacher (EMA copy, no gradients)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # center for teacher outputs
        self.register_buffer("center", torch.zeros(1, head_out_dim))

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of the teacher from student weights."""
        m = self.momentum
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update the center used for teacher output centering."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, views):
        """Forward pass for DINO.

        Args:
            views: list of 2 augmented views, each (B, C, T, H, W).

        Returns:
            dict with keys:
                - "loss": DINO cross-entropy loss (scalar).
                - "student_output": student logits for the first view.
                - "teacher_output": teacher logits for the first view.
        """
        # student forward on all views
        student_outputs = []
        for v in views:
            feat = self.student_backbone(v)
            student_outputs.append(self.student_head(feat))

        # teacher forward on all views (no grad)
        teacher_outputs = []
        with torch.no_grad():
            for v in views:
                feat = self.teacher_backbone(v)
                out = self.teacher_head(feat)
                teacher_outputs.append(out)

        # compute loss: each student view learns from each teacher view
        # (excluding same-view pairs)
        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_out in enumerate(teacher_outputs):
            t_out_centered = t_out - self.center
            t_probs = F.softmax(t_out_centered / self.teacher_temp, dim=-1)
            for s_idx, s_out in enumerate(student_outputs):
                if s_idx == t_idx:
                    continue
                s_log_probs = F.log_softmax(s_out / self.student_temp, dim=-1)
                total_loss += -torch.sum(t_probs * s_log_probs, dim=-1).mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # update center with all teacher outputs
        with torch.no_grad():
            all_teacher = torch.cat(teacher_outputs, dim=0)
            self.update_center(all_teacher)

        return {
            "loss": total_loss,
            "student_output": student_outputs[0],
            "teacher_output": teacher_outputs[0],
        }

    def get_encoder(self):
        """Return the student backbone for downstream use."""
        return self.student_backbone
