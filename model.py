"""
model.py
--------
Regression-only point cloud transformer.

Architecture (per lecturer discussion):
  * Linear embedding of (x, y, intensity) into embed_dim
  * N pre-norm attention blocks:
        h = h + Attention(LayerNorm(h))
        h = h + FFN(LayerNorm(h))          (FFN: up-project -> ReLU -> down-project)
  * Two SEPARATE prediction heads:
        vector_head : (N, 2)  displacement vector to circle center
        alpha_head  : (N,)    normalized signed distance to boundary

There is no classification head and no sigmoid/logit anywhere.
Inside/outside is inferred from sign(alpha) at evaluation time only.

Also includes the NoEmbeddingTransformer variant (professor's earlier
suggestion: Q, K, V applied directly to the raw 3D inputs) for the
embedding-ablation experiment.
"""

import torch
import torch.nn as nn


class PreNormAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention block with residuals."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, h):
        a = self.attn_norm(h)
        attn_out, _ = self.attention(a, a, a, need_weights=False)
        h = h + attn_out
        h = h + self.ffn(self.ffn_norm(h))
        return h


class RegressionPointTransformer(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=1, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [PreNormAttentionBlock(embed_dim, num_heads)
             for _ in range(num_layers)]
        )
        # Separate heads — pure regression outputs
        self.vector_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        v = self.vector_head(h)                 # (B, N, 2)
        alpha = self.alpha_head(h).squeeze(-1)  # (B, N)
        return v, alpha


class NoEmbeddingFirstBlock(nn.Module):
    """First attention layer applied directly to raw 3D input
    (professor's suggestion for the ablation). V maps to embed_dim
    so subsequent processing happens in the high-dimensional space.
    No residual from x (dimension mismatch 3 vs embed_dim)."""

    def __init__(self, input_dim=3, embed_dim=128):
        super().__init__()
        self.W_Q = nn.Linear(input_dim, embed_dim)
        self.W_K = nn.Linear(input_dim, embed_dim)
        self.W_V = nn.Linear(input_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        attn = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        h = self.norm1(torch.matmul(attn, V))
        h = self.norm2(h + self.ffn(h))
        return h


class NoEmbeddingTransformer(nn.Module):
    """Ablation model: no input embedding before the first attention."""

    def __init__(self, input_dim=3, embed_dim=128, num_layers=1):
        super().__init__()
        self.first_layer = NoEmbeddingFirstBlock(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [PreNormAttentionBlock(embed_dim, num_heads=1)
             for _ in range(num_layers - 1)]
        )
        self.vector_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        h = self.first_layer(x)
        for block in self.blocks:
            h = block(h)
        v = self.vector_head(h)
        alpha = self.alpha_head(h).squeeze(-1)
        return v, alpha


def build_model(model_type="standard", input_dim=3, embed_dim=128,
                num_heads=1, num_layers=1):
    if model_type == "standard":
        return RegressionPointTransformer(
            input_dim=input_dim, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_layers)
    elif model_type == "no_embedding":
        return NoEmbeddingTransformer(
            input_dim=input_dim, embed_dim=embed_dim, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
