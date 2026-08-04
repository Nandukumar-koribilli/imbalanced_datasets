"""
Masked Autoencoder (MAE) for 1-D sensor time-series.

The architecture follows the MAE paper (He et al., 2022) adapted for
1-D multi-channel signals such as accelerometer / gyroscope windows:

    1. Non-overlapping 1-D patches → linear projection → positional embedding
    2. Random masking of a high proportion (default 75%) of patches
    3. Transformer encoder operates ONLY on visible patches (efficient)
    4. Lightweight Transformer decoder reconstructs from visible + mask tokens
    5. MSE loss computed ONLY on the masked patches

After pre-training the decoder is discarded.  `MAEEncoder` wraps the
encoder for downstream fine-tuning, producing a 256-d representation
vector compatible with the existing `SHAR_Classifier`.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed1D(nn.Module):
    """Split a (B, C, L) signal into non-overlapping patches and project."""

    def __init__(self, in_channels: int, seq_len: int, patch_size: int = 8,
                 embed_dim: int = 128):
        super().__init__()
        assert seq_len % patch_size == 0, (
            f"seq_len ({seq_len}) must be divisible by patch_size ({patch_size})")
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) raw sensor signal.
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        B, C, L = x.shape
        # Reshape to (B, num_patches, C * patch_size)
        x = x.reshape(B, C, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block (self-attention + FFN)."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings (fixed, not learned)
# ---------------------------------------------------------------------------

def sinusoidal_pos_embed(num_positions: int, dim: int) -> torch.Tensor:
    """Generate sinusoidal positional embeddings (1, num_positions, dim)."""
    pe = torch.zeros(num_positions, dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, N, D)


# ---------------------------------------------------------------------------
# MAE for 1-D signals
# ---------------------------------------------------------------------------

class MAE1D(nn.Module):
    """
    Masked Autoencoder for 1-D time-series (pre-training only).

    After pre-training, discard the decoder and use `MAEEncoder` for
    fine-tuning.
    """

    def __init__(
        self,
        in_channels: int = 9,
        seq_len: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.75,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # --- Encoder ---
        self.patch_embed = PatchEmbed1D(in_channels, seq_len, patch_size,
                                        embed_dim)
        num_patches = self.patch_embed.num_patches

        # Fixed sinusoidal pos-embed (not masked-out, added before masking)
        self.register_buffer(
            "pos_embed", sinusoidal_pos_embed(num_patches, embed_dim)
        )

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, dropout=dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.register_buffer(
            "decoder_pos_embed",
            sinusoidal_pos_embed(num_patches, decoder_embed_dim),
        )

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads, dropout=dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Predict the raw patch pixels (C * patch_size values per patch)
        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      in_channels * patch_size)

    # -- helpers --

    def _random_masking(self, x: torch.Tensor):
        """
        Per-sample random masking by shuffling patch indices.

        Args:
            x: (B, N, D) embedded patches with positional encoding.
        Returns:
            x_visible:   (B, N_vis, D)
            mask:        (B, N)  — 1 = masked, 0 = visible
            ids_restore: (B, N)  — indices to un-shuffle
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - self.mask_ratio))

        # Random noise per sample → argsort gives random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        # Keep the first num_keep tokens (in shuffled order)
        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Binary mask: 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) → (B, num_patches, C * patch_size)"""
        B, C, L = x.shape
        N = self.patch_embed.num_patches
        P = self.patch_size
        x = x.reshape(B, C, N, P).permute(0, 2, 1, 3).reshape(B, N, C * P)
        return x

    # -- forward --

    def forward_encoder(self, x: torch.Tensor):
        """Encode only the visible patches."""
        x = self.patch_embed(x)       # (B, N, D)
        x = x + self.pos_embed        # add positional embeddings

        # Mask
        x, mask, ids_restore = self._random_masking(x)

        # Transformer encoder
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x_enc: torch.Tensor,
                        ids_restore: torch.Tensor):
        """Decode: insert mask tokens and reconstruct full sequence."""
        # Project encoder output to decoder dimension
        x = self.decoder_embed(x_enc)   # (B, N_vis, D_dec)

        B, N_vis, D = x.shape
        N = self.patch_embed.num_patches

        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)       # (B, N, D_dec)

        # Un-shuffle to original order
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # Add decoder positional embeddings
        x_full = x_full + self.decoder_pos_embed

        # Transformer decoder
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)

        # Predict patch values
        pred = self.decoder_pred(x_full)   # (B, N, C*P)
        return pred

    def forward(self, x: torch.Tensor):
        """
        Full MAE forward: encode → decode → loss.

        Returns:
            loss:  scalar MSE on masked patches only.
            pred:  (B, N, C*P) predicted patches.
            mask:  (B, N) binary mask.
        """
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)

        # Target: patchified raw input
        target = self._patchify(x)

        # MSE loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)               # per-patch MSE → (B, N)
        loss = (loss * mask).sum() / mask.sum() # average over masked patches

        return loss, pred, mask


# ---------------------------------------------------------------------------
# Encoder wrapper for downstream fine-tuning
# ---------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """
    Encoder-only wrapper used after MAE pre-training.

    Runs the Transformer encoder on **all** patches (no masking) and
    produces a fixed-size representation via global average pooling →
    linear projection.  Output dimension = ``rep_dim`` (default 256),
    matching ``SHAREncoder`` so the same ``SHAR_Classifier`` head works.
    """

    def __init__(
        self,
        in_channels: int = 9,
        seq_len: int = 128,
        patch_size: int = 8,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        rep_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed1D(in_channels, seq_len, patch_size,
                                        embed_dim)
        num_patches = self.patch_embed.num_patches

        self.register_buffer(
            "pos_embed", sinusoidal_pos_embed(num_patches, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads, dropout=dropout)
            for _ in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Projection to representation space
        self.fc = nn.Linear(embed_dim, rep_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) sensor signal.
        Returns:
            (B, rep_dim) representation vector.
        """
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Global average pooling over patch dimension
        x = x.mean(dim=1)             # (B, embed_dim)
        x = self.relu(self.fc(x))     # (B, rep_dim)
        return x


# ---------------------------------------------------------------------------
# Utility: transfer encoder weights from MAE1D → MAEEncoder
# ---------------------------------------------------------------------------

def transfer_mae_weights(mae_model: MAE1D, encoder: MAEEncoder):
    """
    Copy the pre-trained encoder weights from a full MAE1D model into a
    standalone MAEEncoder.  Keys are matched by stripping the prefix
    difference (``patch_embed.`` / ``encoder_blocks.`` → ``patch_embed.`` /
    ``blocks.``).
    """
    mae_sd = mae_model.state_dict()
    enc_sd = encoder.state_dict()

    mapping = {}
    for k in enc_sd:
        if k.startswith("blocks."):
            src_k = "encoder_" + k          # encoder_blocks.*
        elif k == "norm.weight":
            src_k = "encoder_norm.weight"
        elif k == "norm.bias":
            src_k = "encoder_norm.bias"
        else:
            src_k = k                       # patch_embed.*, pos_embed, fc.*, relu (no weight)

        if src_k in mae_sd:
            mapping[k] = mae_sd[src_k]

    # fc / relu layers won't exist in MAE1D — leave randomly initialised
    enc_sd.update(mapping)
    encoder.load_state_dict(enc_sd)
    return encoder


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C, L = 4, 9, 128

    print("=== MAE1D (pre-training) ===")
    mae = MAE1D(in_channels=C, seq_len=L)
    x = torch.randn(B, C, L)
    loss, pred, mask = mae(x)
    print(f"  Input:  {x.shape}")
    print(f"  Loss:   {loss.item():.4f}")
    print(f"  Pred:   {pred.shape}")
    print(f"  Mask:   {mask.shape}  (masked ratio: {mask.float().mean():.2f})")

    print("\n=== MAEEncoder (fine-tuning) ===")
    enc = MAEEncoder(in_channels=C, seq_len=L)
    transfer_mae_weights(mae, enc)
    rep = enc(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {rep.shape}")

    print("\n=== Compatibility with SHAR_Classifier ===")
    from shar_model import SHAR_Classifier
    clf = SHAR_Classifier(enc, num_classes=6)
    logits = clf(x)
    print(f"  Logits: {logits.shape}")
