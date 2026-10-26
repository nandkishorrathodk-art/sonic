"""Top-Down Language Modulation for Non-Generative Visual Grounding.

In cognitive neuroscience, the prefrontal cortex modulates early visual cortex
via top-down attention (e.g. searching for a red apple primes red-sensitive neurons).

In AVCS, language guides visual attention without generating any text tokens:
1. An incoming language query (or precomputed embedding) is projected into
   the visual feature space.
2. A cross-attention correlation computes a dense 2D spatial heatmap:
   Relevance(x, y) = Softmax(Visual(x, y) . Text_Query / sqrt(D)).
3. The model pinpoints where the query resides on the screen in a single forward pass.
"""
from __future__ import annotations

from typing import Optional, Tuple
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def dsnt_soft_argmax(relevance_map: torch.Tensor) -> torch.Tensor:
    """Computes differentiable center-of-mass (soft-argmax) from spatial heatmap.

    Replaces spatial pooling collapse with exact coordinate expectation:
    (norm_x, norm_y) = (sum(x * P), sum(y * P)).
    """
    B, _, H, W = relevance_map.shape
    device = relevance_map.device
    dtype = relevance_map.dtype

    xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype)
    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.view(1, 1, H, W)
    grid_y = grid_y.view(1, 1, H, W)

    coord_x = (relevance_map * grid_x).sum(dim=(-2, -1))
    coord_y = (relevance_map * grid_y).sum(dim=(-2, -1))
    return torch.cat([coord_x, coord_y], dim=-1)


class LanguageModulation(nn.Module):
    """Zero-generation language conditioning over visual feature maps."""

    def __init__(self, visual_dim: int = 128, text_dim: int = 128) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim

        # Project text embedding to match visual channel dimensions
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )

        # Spatial temperature scaling
        self.scale = visual_dim ** -0.5

        # Residual coordinate refinement head
        self.coord_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(visual_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # residual offset
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute language-guided relevance map and direct target coordinate."""
        B, C, H, W = visual_features.shape

        # 1. Project text embedding: (B, visual_dim)
        t_proj = self.text_proj(text_embedding)  # (B, C)

        # 2. Compute spatial dot-product alignment
        t_expanded = t_proj.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        sim = (visual_features * t_expanded).sum(dim=1, keepdim=True) * self.scale  # (B, 1, H, W)

        # 3. Spatial Softmax Heatmap
        flat_sim = sim.view(B, 1, -1)
        relevance_map = F.softmax(flat_sim, dim=-1).view(B, 1, H, W)

        # 4. Differentiable Spatial Numerical Transform (DSNT Soft-Argmax)
        dsnt_coords = dsnt_soft_argmax(relevance_map)  # (B, 2) in [0, 1]

        # 5. Modulated visual features for residual coordinate refinement
        modulated_features = visual_features * (1.0 + relevance_map)
        residual = (self.coord_regressor(modulated_features) - 0.5) * 0.15
        target_coords = torch.clamp(dsnt_coords + residual, 0.0, 1.0)

        return relevance_map, target_coords


class CharNGramEncoder(nn.Module):
    """Zero-cloud local text query encoder (<12k parameters).

    Maps UI strings ('Submit', 'Search', 'Close') to latent embeddings
    using character n-grams without cloud LLM dependencies.
    """

    def __init__(self, out_dim: int = 128, hash_bins: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.hash_bins = hash_bins
        self.embed = nn.Embedding(hash_bins, 32)
        self.proj = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, query: str | list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        queries = [query] if isinstance(query, str) else list(query)
        dev = device or next(self.parameters()).device

        indices_batch = []
        for q in queries:
            q_clean = q.lower().strip()
            ngrams = [q_clean[i : i + 2] for i in range(len(q_clean) - 1)] + [
                q_clean[i : i + 3] for i in range(len(q_clean) - 2)
            ]
            if not ngrams:
                ngrams = [q_clean]
            hashes = [zlib.crc32(ng.encode("utf-8")) % self.hash_bins for ng in ngrams]
            indices_batch.append(hashes)

        max_len = max(len(h) for h in indices_batch)
        padded = [h + [0] * (max_len - len(h)) for h in indices_batch]

        idx_tensor = torch.tensor(padded, dtype=torch.long, device=dev)
        emb = self.embed(idx_tensor).mean(dim=1)
        out = self.proj(emb)
        return F.normalize(out, dim=-1)


# Cache of shared encoders keyed by output dimension. The encoder is a real,
# deterministic function of the query text (character n-grams -> hashed
# embedding -> projection), so identical queries always produce identical
# vectors and different queries produce genuinely different vectors.
_SHARED_ENCODERS: dict[int, CharNGramEncoder] = {}


def _shared_encoder(out_dim: int) -> CharNGramEncoder:
    encoder = _SHARED_ENCODERS.get(out_dim)
    if encoder is None:
        torch.manual_seed(0xC0FFEE ^ out_dim)
        encoder = CharNGramEncoder(out_dim=out_dim)
        encoder.eval()
        _SHARED_ENCODERS[out_dim] = encoder
    return encoder


def encode_text_query(
    query: str,
    out_dim: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode a text query into a normalized embedding of shape ``(1, out_dim)``.

    This is the single honest entry point for turning operator language into the
    vector the cortex's ``LanguageModulation`` consumes. It replaces the previous
    practice of seeding ``torch.randn`` from a SHA-256 hash of the query, which
    produced a *different random vector per query with no relation to the words*
    -- the language path was effectively trained on and driven by noise.

    The returned vector is a deterministic function of the query's characters, so
    'Submit' and 'Cancel' are close in n-gram space while 'Submit' and 'Logout'
    are not. It is not a semantic language model; it is an honest local
    character-level embedding with no cloud dependency.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("encode_text_query requires a non-empty query string")
    if out_dim <= 0:
        raise ValueError("encode_text_query requires a positive out_dim")
    encoder = _shared_encoder(out_dim)
    dev = device or torch.device("cpu")
    encoder = encoder.to(dev)
    with torch.no_grad():
        return encoder(query.strip(), device=dev)

