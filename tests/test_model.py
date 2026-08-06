"""Model architecture tests — skipped automatically when torch is unavailable."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from src.shar_model import SHAREncoder, SHAR_Pretrain, SHAR_Classifier  # noqa: E402
from src.random_masking import apply_random_masking_batch  # noqa: E402


def test_encoder_output_dim():
    enc = SHAREncoder(in_channels=9, seq_len=128)
    out = enc(torch.randn(4, 9, 128))
    assert out.shape == (4, 256)


def test_pretrain_projector_dim():
    model = SHAR_Pretrain(in_channels=9, seq_len=128)
    z = model(torch.randn(4, 9, 128))
    assert z.shape == (4, 128)


def test_classifier_logits_uci_har():
    enc = SHAREncoder(in_channels=9, seq_len=128)
    assert SHAR_Classifier(enc, num_classes=6)(torch.randn(2, 9, 128)).shape == (2, 6)


def test_random_masking_zeroes_but_preserves_shape():
    x = torch.ones(8, 9, 128)
    masked = apply_random_masking_batch(x, mask_prob=0.2)
    assert masked.shape == x.shape
    frac_zeroed = (masked == 0).float().mean().item()
    assert 0.05 < frac_zeroed < 0.45  # ~20 % expected


def test_pretrained_checkpoint_loads_if_present():
    ckpt = os.path.join(os.path.dirname(__file__), "..", "models", "shar_encoder_pretrained.pth")
    if not os.path.exists(ckpt):
        pytest.skip("no checkpoint on disk")
    enc = SHAREncoder(in_channels=9, seq_len=128)
    enc.load_state_dict(torch.load(ckpt, map_location="cpu"))  # strict=True
