"""
PyTorch Mixture Density Network (MDN) for multimodal probabilistic forecasting of EV charging session duration and required energy.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def mdn_loss_function(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    rho: Optional[torch.Tensor] = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Calculates the Negative Log-Likelihood (NLL) loss for a Gaussian Mixture Model.

    Supports both full bivariate covariance (with correlation rho) and diagonal covariance.

    Args:
        pi: Mixture component probabilities [batch_size, num_mixtures].
        mu: Mixture component means [batch_size, num_mixtures, output_dim].
        sigma: Mixture component standard deviations [batch_size, num_mixtures, output_dim].
        target: Target ground truth tensor [batch_size, output_dim].
        rho: Optional mixture component correlation coefficient [batch_size, num_mixtures].
        eps: Small constant for numerical stability.

    Returns:
        torch.Tensor: Mean NLL loss scalar across batch.
    """
    # Numerical safety clamp on standard deviation
    sigma_clamped = torch.clamp(sigma, min=1e-3, max=1e2)
    batch_size, num_mixtures, output_dim = mu.shape
    target_expanded = target.unsqueeze(1).expand_as(mu)

    log_2pi = torch.log(torch.tensor(2.0 * 3.141592653589793, device=target.device, dtype=target.dtype))

    # Bivariate full covariance formulation with correlation parameter rho
    if output_dim == 2 and rho is not None:
        rho_clamped = torch.clamp(rho, min=-0.99, max=0.99)
        z1 = (target_expanded[:, :, 0] - mu[:, :, 0]) / sigma_clamped[:, :, 0]
        z2 = (target_expanded[:, :, 1] - mu[:, :, 1]) / sigma_clamped[:, :, 1]
        
        one_minus_rho_sq = 1.0 - rho_clamped**2
        quad_form = (z1**2 - 2.0 * rho_clamped * z1 * z2 + z2**2) / torch.clamp(one_minus_rho_sq, min=1e-4)
        
        log_gaussian_sum = -log_2pi - torch.log(sigma_clamped[:, :, 0]) - torch.log(sigma_clamped[:, :, 1]) \
                           - 0.5 * torch.log(torch.clamp(one_minus_rho_sq, min=1e-4)) - 0.5 * quad_form
    else:
        # Diagonal covariance formulation across output dimensions
        log_gaussian = -0.5 * (
            log_2pi + 2.0 * torch.log(sigma_clamped) + ((target_expanded - mu) / sigma_clamped) ** 2
        )
        log_gaussian_sum = torch.sum(log_gaussian, dim=2)  # [batch_size, num_mixtures]

    # Combine with mixture log weights: log(pi_k * N(y|mu_k, Sigma_k)) = log(pi_k) + log_N
    log_pi = torch.log(torch.clamp(pi, min=eps))
    weighted_log_prob = log_pi + log_gaussian_sum  # [batch_size, num_mixtures]

    # Stable summation across mixture components via logsumexp
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
        # 4. Correlation coefficient head for bivariate covariance (K components)
        if self.output_dim == 2:
            self.rho_head = nn.Linear(in_dim, num_mixtures)
        else:
            self.rho_head = None

    def forward(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass through backbone and GMM heads.

        Args:
            x: Input feature tensor [batch_size, input_dim].

        Returns:
            Tuple[torch.Tensor, ...]: (pi, mu, sigma) or (pi, mu, sigma, rho) parameters.
        """
        features = self.backbone(x)

        # Softmax over components to ensure sum(pi_k) = 1
        pi = F.softmax(self.pi_head(features), dim=-1)

        # Reshape mu and sigma to [batch_size, num_mixtures, output_dim]
        mu = self.mu_head(features).view(-1, self.num_mixtures, self.output_dim)

        # Softplus activation + clamping to enforce numerical stability
        sigma_raw = self.sigma_head(features).view(-1, self.num_mixtures, self.output_dim)
        sigma = torch.clamp(F.softplus(sigma_raw) + 1e-4, min=1e-3, max=1e2)

        if self.rho_head is not None:
            rho = 0.99 * torch.tanh(self.rho_head(features))  # [batch_size, num_mixtures]
            return pi, mu, sigma, rho

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
            outputs = self.forward(x)
            pi, mu, sigma = outputs[0], outputs[1], outputs[2]

            # Expected Mean E[Y] = sum_k (pi_k * mu_k)
            pi_expanded = pi.unsqueeze(-1)
            mean = torch.sum(pi_expanded * mu, dim=1)  # [batch_size, output_dim]

            # Variance Var[Y] = sum_k (pi_k * (sigma_k^2 + mu_k^2)) - (E[Y])^2
            variance_terms = torch.sum(pi_expanded * (sigma**2 + mu**2), dim=1)
            var = variance_terms - mean**2
            std = torch.sqrt(torch.clamp(var, min=1e-6))

        return mean, std

    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """Generates Monte Carlo samples from predicted GMM distribution.

        Args:
            x: Feature tensor [batch_size, input_dim].
            num_samples: Number of samples per input instance.

        Returns:
            torch.Tensor: Samples tensor [batch_size, num_samples, output_dim].
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            pi, mu, sigma = outputs[0], outputs[1], outputs[2]
            rho = outputs[3] if len(outputs) > 3 else None

            cat_dist = torch.distributions.Categorical(probs=pi)
            indices = cat_dist.sample((num_samples,)).transpose(0, 1)  # [batch_size, num_samples]

            indices_exp = indices.unsqueeze(-1).expand(-1, -1, self.output_dim)  # [batch_size, num_samples, output_dim]
            sampled_mu = torch.gather(mu, dim=1, index=indices_exp)
            sampled_sigma = torch.gather(sigma, dim=1, index=indices_exp)

            if self.output_dim == 2 and rho is not None:
                indices_rho = indices.unsqueeze(-1)  # [batch_size, num_samples, 1]
                sampled_rho = torch.gather(rho.unsqueeze(-1), dim=1, index=indices_rho).squeeze(-1)
                
                z1 = torch.randn_like(sampled_mu[:, :, 0])
                z2 = torch.randn_like(sampled_mu[:, :, 1])
                
                s1 = sampled_mu[:, :, 0] + sampled_sigma[:, :, 0] * z1
                s2 = sampled_mu[:, :, 1] + sampled_sigma[:, :, 1] * (sampled_rho * z1 + torch.sqrt(torch.clamp(1.0 - sampled_rho**2, min=1e-4)) * z2)
                return torch.stack([s1, s2], dim=-1)

            samples = torch.normal(sampled_mu, sampled_sigma)
            return samples
