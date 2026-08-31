"""
Unit tests for Mixture Density Network (MDN) and Quantile TCN architectures and loss functions.
"""

import torch
import pytest

from src.models.mdn_network import MixtureDensityNetwork, mdn_loss_function
from src.models.quantile_tcn import QuantileTCN, pinball_loss


def test_mdn_forward_and_loss():
    batch_size = 16
    input_dim = 10
    num_mixtures = 3
    output_dim = 2

    model = MixtureDensityNetwork(
        input_dim=input_dim,
        hidden_dims=[32, 32],
        num_mixtures=num_mixtures,
        output_dim=output_dim,
    )

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)

    outputs = model(x)
    pi, mu, sigma = outputs[0], outputs[1], outputs[2]
    rho = outputs[3] if len(outputs) > 3 else None

    assert pi.shape == (batch_size, num_mixtures)
    assert mu.shape == (batch_size, num_mixtures, output_dim)
    assert sigma.shape == (batch_size, num_mixtures, output_dim)
    if rho is not None:
        assert rho.shape == (batch_size, num_mixtures)
        assert (rho >= -1.0).all() and (rho <= 1.0).all()

    # Check softmax sum to 1
    assert torch.allclose(pi.sum(dim=-1), torch.ones(batch_size), atol=1e-4)
    # Check strict positivity of sigma
    assert (sigma > 0).all()

    loss = mdn_loss_function(pi, mu, sigma, y, rho=rho)
    assert not torch.isnan(loss)
    assert loss.dim() == 0  # Scalar loss


def test_mdn_prediction_and_sampling():
    model = MixtureDensityNetwork(input_dim=8, num_mixtures=4, output_dim=2)
    x = torch.randn(5, 8)

    mean, std = model.predict_distribution(x)
    assert mean.shape == (5, 2)
    assert std.shape == (5, 2)

    samples = model.sample(x, num_samples=20)
    assert samples.shape == (5, 20, 2)


def test_quantile_tcn_forward_and_loss():
    batch_size = 8
    input_dim = 6
    sequence_length = 24
    quantiles = [0.1, 0.5, 0.9]

    model = QuantileTCN(
        input_dim=input_dim,
        num_channels=[16, 16],
        kernel_size=3,
        quantiles=quantiles,
    )

    x = torch.randn(batch_size, input_dim, sequence_length)
    y_true = torch.randn(batch_size, 1, sequence_length)

    out = model(x)
    assert out.shape == (batch_size, len(quantiles), sequence_length)

    # Check quantile crossing prevention (q_0.1 <= q_0.5 <= q_0.9)
    assert (out[:, 0, :] <= out[:, 1, :]).all()
    assert (out[:, 1, :] <= out[:, 2, :]).all()

    loss = pinball_loss(out, y_true, quantiles=quantiles)
    assert not torch.isnan(loss)
    assert loss.dim() == 0


def test_mdn_default_input_dim():
    model = MixtureDensityNetwork()
    assert model.input_dim == 12


def test_trainer_auto_device():
    from src.models.trainer import ModelTrainer
    model = MixtureDensityNetwork(input_dim=12)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = ModelTrainer(model=model, loss_fn=mdn_loss_function, optimizer=optimizer, device="auto")
    assert isinstance(trainer.device, torch.device)
