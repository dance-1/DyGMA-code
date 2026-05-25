import math

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError("DyGMA requires mamba-ssm. Install it with `pip install mamba-ssm`.") from exc


class DyGMAModel(nn.Module):
    """Dynamic dual-graph model for multivariate time-series anomaly detection."""

    def __init__(self, win_size, num_vars, patch_len=1, d_model=128, d_state=16, d_e=64):
        super(DyGMAModel, self).__init__()
        self.win_size = win_size
        self.num_vars = num_vars
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = patch_len

        self.patch_embedding = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

        self.norm = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.node_embeddings = nn.Parameter(torch.randn(num_vars, d_e))

        self.reconstruct_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, self.patch_len),
        )

    def forward(self, x):
        B, L, D = x.shape

        # Channel-independent temporal encoding.
        x_ci = x.permute(0, 2, 1).contiguous().reshape(B * D, 1, L)
        x_patch = self.patch_embedding(x_ci)
        x_patch = x_patch.transpose(1, 2).contiguous()
        num_patches = x_patch.shape[1]

        mamba_out = self.mamba(x_patch)

        # Use pooled temporal features for graph construction.
        temporal_feat = mamba_out.mean(dim=1)
        spatial_feat_for_graph = temporal_feat.reshape(B, D, self.d_model)
        spatial_feat_for_graph = self.norm(spatial_feat_for_graph)

        # Keep patch-level features for reconstruction.
        spatial_feat_full = mamba_out.reshape(B, D, num_patches, self.d_model)
        spatial_feat_full = self.norm(spatial_feat_full)

        # Transient graph from current-window features.
        Q = self.W_q(spatial_feat_for_graph)
        K = self.W_k(spatial_feat_for_graph)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        S_series = torch.softmax(scores, dim=-1)

        # Learnable prior graph shared across windows.
        E = self.node_embeddings
        prior_scores = torch.matmul(E, E.transpose(0, 1)) / math.sqrt(E.shape[1])
        P_prior = torch.softmax(prior_scores, dim=-1)
        P_prior = P_prior.unsqueeze(0).expand(B, -1, -1)

        # Single-step graph routing followed by patch reconstruction.
        V_full = self.W_v(spatial_feat_full)
        Z_full = torch.einsum("bij,bjkd->bikd", S_series, V_full)
        rec_out_patches = self.reconstruct_decoder(Z_full)
        rec_out = rec_out_patches.reshape(B, D, -1)
        rec_out = rec_out.transpose(1, 2).contiguous()

        return rec_out, S_series, P_prior
