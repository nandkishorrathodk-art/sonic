import pytest
import torch

from avcs.training.vicreg_safeguard import normalize_embedding, vicreg_variance_loss


def test_normalized_embedding_has_unit_norm():
    x = torch.randn(16, 256)
    z = normalize_embedding(x)
    assert z.shape == x.shape
    assert torch.allclose(torch.linalg.vector_norm(z, dim=-1), torch.ones(16), atol=1e-5)


def test_variance_loss_is_finite_scalar_and_backpropagates():
    x = torch.randn(32, 256, requires_grad=True)
    loss = vicreg_variance_loss(x, target_std=1.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_low_variance_batch_is_penalized():
    # Small, non-identical values avoid a perfectly flat tensor's zero-gradient edge case.
    x = torch.full((16, 8), 0.2) + torch.randn(16, 8) * 1e-3
    loss = vicreg_variance_loss(x, target_std=1.0)
    assert loss.item() > 0.9


def test_batch_size_one_is_rejected():
    with pytest.raises(ValueError, match="batch_size >= 2"):
        vicreg_variance_loss(torch.randn(1, 256))


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
        torch.tensor([[float("inf"), 0.0], [0.0, 1.0]]),
    ],
)
def test_nonfinite_values_are_rejected(bad):
    with pytest.raises(ValueError, match="NaN or Inf"):
        vicreg_variance_loss(bad)


def test_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError, match=r"\[batch, embedding_dim\]"):
        vicreg_variance_loss(torch.randn(16))
