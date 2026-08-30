"""
PyTorch Mixture Density Network (MDN) for multimodal probabilistic forecasting of EV charging session duration and required energy.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mdn_loss_function(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Calculates the Negative Log-Likelihood (NLL) loss for a Gaussian Mixture Model.

    Args:
        pi: Mixture component probabilities [batch_size, num_mixtures].
        mu: Mixture component means [batch_size, num_mixtures, output_dim].
        sigma: Mixture component standard deviations [batch_size, num_mixtures, output_dim].
        target: Target ground truth tensor [batch_size, output_dim].
        eps: Small constant for numerical stability.

    Returns:
        torch.Tensor: Mean NLL loss scalar across batch.
    """
    # Expand target to match mixtures dimension: [batch_size, 1, output_dim]
    target_expanded = target.unsqueeze(1).expand_as(mu)

    # Compute Gaussian NLL per component and per output dimension
    # log prob = -0.5 * log(2 * pi) - log(sigma) - 0.5 * ((y - mu) / sigma)^2
    log_2pi = torch.log(torch.tensor(2.0 * 3.141592653589793, device=target.device))
    log_gaussian = -0.5 * (
        log_2pi + 2.0 * torch.log(sigma + eps) + ((target_expanded - mu) / (sigma + eps)) ** 2
    )

    # Sum log probabilities across output dimensions (assuming independence conditional on component k)
    log_gaussian_sum = torch.sum(log_gaussian, dim=2)  # [batch_size, num_mixtures]

    # Combine with mixture log weights: log(pi_k * N(y|mu_k, sigma_k)) = log(pi_k) + log_N
    log_pi = torch.log(pi + eps)
    weighted_log_prob = log_pi + log_gaussian_sum  # [batch_size, num_mixtures]

    # Log-Sum-Exp trick for stable summation across mixture components
    log_likelihood = torch.logsumexp(weighted_log_prob, dim=1)  # [batch_size]

    return -torch.mean(log_likelihood)


class MixtureDensityNetwork(nn.Module):
    """PyTorch Mixture Density Network combining MLP backbone with Gaussian Mixture output heads."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: List[int] = None,
        num_mixtures: int = 5,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initializes MDN architecture.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            num_mixtures: Number of Gaussian components (K).
            output_dim: Dimension of continuous targets (e.g. [duration, energy]).
            dropout: Dropout probability.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        self.input_dim = input_dim
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

        # Feature Extractor Backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Output Heads
        # 1. Mixture weights pi (K components)
        self.pi_head = nn.Linear(in_dim, num_mixtures)
        # 2. Means mu (K * output_dim)
        self.mu_head = nn.Linear(in_dim, num_mixtures * output_dim)
        # 3. Standard deviations sigma (K * output_dim)
        self.sigma_head = nn.Linear(in_dim, num_mixtures * output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through backbone and GMM heads.

        Args:
            x: Input feature tensor [batch_size, input_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (pi, mu, sigma) parameters.
        """
        features = self.backbone(x)

        # Softmax over components to ensure sum(pi_k) = 1
        pi = F.softmax(self.pi_head(features), dim=-1)

        # Reshape mu and sigma to [batch_size, num_mixtures, output_dim]
        mu = self.mu_head(features).view(-1, self.num_mixtures, self.output_dim)

        # Softplus activation for sigma to enforce strict positivity (sigma > 0)
        sigma_raw = self.sigma_head(features).view(-1, self.num_mixtures, self.output_dim)
        sigma = F.softplus(sigma_raw) + 1e-4

        return pi, mu, sigma

    def predict_distribution(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes expected mean vector and standard deviation vector from predicted GMM parameters.

        Args:
            x: Input feature tensor [batch_size, input_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Expected mean [batch_size, output_dim] and total std [batch_size, output_dim].
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)

            # Expected Mean E[Y] = sum_k (pi_k * mu_k)
            # pi shape: [batch_size, num_mixtures, 1]
            pi_expanded = pi.unsqueeze(-1)
            mean = torch.sum(pi_expanded * mu, dim=1)  # [batch_size, output_dim]

            # Variance Var[Y] = sum_k (pi_k * (sigma_k^2 + mu_k^2)) - (E[Y])^2
            variance_terms = torch.sum(pi_expanded * (sigma**2 + mu**2), dim=1)
            var = variance_terms - mean**2
            std = torch.sqrt(torch.clamp(var, min=1e-6))

        return mean, std

    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """Generates Monte Carlo samples from predicted GMM distribution using vectorized sampling.

        Args:
            x: Feature tensor [batch_size, input_dim].
            num_samples: Number of samples per input instance.

        Returns:
            torch.Tensor: Samples tensor [batch_size, num_samples, output_dim].
        """
        self.eval()
        with torch.no_grad():
            pi, mu, sigma = self.forward(x)

            # Vectorized Categorical mixture selection
            cat_dist = torch.distributions.Categorical(probs=pi)
            indices = cat_dist.sample((num_samples,)).transpose(0, 1)  # [batch_size, num_samples]

            # Expand indices for output dimensions and gather components
            indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.output_dim)  # [batch_size, num_samples, output_dim]
            sampled_mu = torch.gather(mu, dim=1, index=indices_exp)
            sampled_sigma = torch.gather(sigma, dim=1, index=indices_exp)

            # Vectorized normal sampling
            samples = torch.normal(sampled_mu, sampled_sigma)
            return samples
