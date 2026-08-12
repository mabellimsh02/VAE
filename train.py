"""
Trains the VAE from model.py on MNIST digits, then uses the trained model
to generate new digit images.

The training loss has two parts that pull in different directions:
  1. Reconstruction loss -- how different is the decoded image from the
     original? This pushes the model to encode/decode accurately.
  2. KL-divergence -- how far is the latent distribution (mu, sigma) from
     a standard normal distribution? This keeps the latent space smooth
     and "well organized," which is what lets us sample random points
     from it later and get sensible-looking digits out.

Without the KL term, the model would just become a regular autoencoder
that memorizes exact codes for each image, with gaps in latent space that
decode to garbage. Without the reconstruction term, it would collapse to
just outputting the standard normal distribution and ignore the input
entirely. Training balances both at once.
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import VariationalAutoEncoder

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784  # 28x28 MNIST images, flattened into one vector
H_DIM = 200  # size of the hidden layer between image and latent space
Z_DIM = 20  # size of the latent space (how much we compress each image down to)
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4

# data: ToTensor() converts pixel values from 0-255 ints into 0-1 floats,
# which matches the sigmoid-bounded output of the decoder.
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
# BCELoss with reduction="sum" (rather than the default "mean") pairs
# naturally with the KL term below, since KL is also computed as a sum
# over all elements rather than an average.
loss_fn = nn.BCELoss(reduction="sum")


def train():
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, _) in loop:
            # We only need the images here, not their digit labels (the
            # `_`), since a VAE trains in an unsupervised way -- it never
            # needs to know what digit it's looking at.
            x = x.view(x.shape[0], INPUT_DIM).to(DEVICE)
            x_reconstructed, mu, sigma = model(x)

            # How close is the reconstructed image to the original?
            reconstruction_loss = loss_fn(x_reconstructed, x)

            # How far is the latent distribution (mu, sigma) from a
            # standard normal (mean 0, variance 1)? This has a closed-form
            # solution for two normal distributions, which is why it can
            # be computed directly from mu and sigma without sampling.
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = reconstruction_loss + kl_div

            # standard PyTorch update: clear old gradients, compute new
            # ones via backprop, then step the optimizer.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())


def inference(digit, num_examples=1):
    """
    Generates (num_examples) images of a particular digit by finding one
    reference example of that digit in the dataset, encoding it, then
    sampling new z vectors via the reparameterization trick and decoding.
    """
    # Scan the dataset once to grab exactly one example image per digit
    # (0 through 9), so we have a reference point in latent space for
    # each digit to sample around.
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    # Encode each reference image into its (mu, sigma) latent distribution.
    # No gradients are needed here since we're just doing inference.
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, INPUT_DIM).to(DEVICE))
        encodings_digit.append((mu, sigma))

    # For the requested digit, repeatedly sample a *different* random z
    # around its (mu, sigma) and decode it. Because each sample uses fresh
    # random noise (epsilon), each output is a distinct but similar-looking
    # variation of that digit.
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")


if __name__ == "__main__":
    train()
    for idx in range(10):
        inference(idx, num_examples=5)
