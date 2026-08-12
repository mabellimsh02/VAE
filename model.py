"""
Variational Autoencoder (VAE) for MNIST digits.

A normal autoencoder learns to compress an image into a small vector
(the "latent code") and decompress it back. A VAE does something similar,
but instead of encoding an image into one fixed point in latent space, it
encodes it into a *probability distribution* (a mean `mu` and a spread
`sigma`). We then sample a point from that distribution to decode.

Why bother with a distribution instead of a single point? Because it
forces nearby points in latent space to decode to similar-looking images.
That's what makes a VAE "generative" -- once trained, we can sample random
points from latent space and decode them into new, never-before-seen
digits that still look like real digits.
"""

import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # Encoder: compresses a flattened image (input_dim) down to a
        # hidden representation (h_dim), then down further to the latent
        # space (z_dim). We need two separate output layers here because
        # the encoder must produce both a mean and a spread for the
        # latent distribution, not just one value per latent dimension.
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # Decoder: mirrors the encoder in reverse, expanding a latent
        # vector (z_dim) back into a hidden representation (h_dim) and
        # then into a full reconstructed image (input_dim).
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # mu and sigma need to be able to take any value (including
        # negative), so no activation function is applied to them --
        # only the shared hidden layer before them gets a ReLU.
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        # Sigmoid squashes the output to (0, 1), matching normalized
        # pixel values, so the result can be compared directly against
        # the original image with a binary cross-entropy loss.
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)

        # Reparameterization trick: we want to sample a random z from
        # the distribution described by (mu, sigma), but sampling
        # directly is a non-differentiable operation, and backprop can't
        # flow through it. Instead we sample noise (epsilon) from a
        # fixed standard normal distribution and shift/scale it by mu and
        # sigma. Now the randomness is isolated in epsilon, and mu/sigma
        # remain part of a differentiable computation the optimizer can
        # learn through.
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon

        x_reconstructed = self.decode(z)
        # mu and sigma are returned too because the training loop needs
        # them to compute the KL-divergence term of the loss, not just
        # the reconstructed image.
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    # Quick sanity check: confirm the shapes flowing through the model
    # are what we expect before wiring it up to real training.
    batch_size = 4
    x = torch.randn(batch_size, 784)
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)  # (4, 784)
    print(mu.shape)  # (4, 20)
    print(sigma.shape)  # (4, 20)
