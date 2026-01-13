# ============================================================
# FASHION / SKETCH ART GENERATOR (DCGAN on Fashion-MNIST)
# - Dataset: Fashion-MNIST (tshirt, trouser, dress, shoe, bag...)
# - Stable DCGAN: BCEWithLogitsLoss, no Sigmoid in D, label smoothing
# - Saves ONLY one final comparison image: real vs generated
# ============================================================

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

# -----------------------------
# Hyperparameters
# -----------------------------
IMAGE_SIZE = 64
NC = 1        # grayscale
NZ = 100      # latent (noise) dimension
NGF = 64      # generator feature maps
NDF = 64      # discriminator feature maps
BATCH_SIZE = 128
NUM_EPOCHS = 20
LR_G = 0.0002
LR_D = 0.0001
BETA1 = 0.5


# -----------------------------
# Weight Initialization (DCGAN)
# -----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: Z (NZ x 1 x 1)
            nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        return self.main(x)


# -----------------------------
# Discriminator (no Sigmoid)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (NC x 64 x 64)
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False)
            # No Sigmoid here (we use BCEWithLogitsLoss)
        )

    def forward(self, x):
        return self.main(x).view(-1)  # (batch_size,)


# -----------------------------
# Data Loader: Fashion-MNIST → 64x64
# -----------------------------
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
    ])

    # Using FashionMNIST instead of MNIST
    dataset = datasets.FashionMNIST(
        root="data_fashion",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -----------------------------
# Training Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training Using:", device)

netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))

dataloader = get_dataloader()


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        bs = imgs.size(0)

        # Label smoothing: real = 0.9, fake = 0.0
        real_label = torch.full((bs,), 0.9, device=device)
        fake_label = torch.full((bs,), 0.0, device=device)

        # ---- Train Discriminator ----
        netD.zero_grad()

        out_real = netD(imgs)
        lossD_real = criterion(out_real, real_label)

        noise = torch.randn(bs, NZ, 1, 1, device=device)
        fake_imgs = netG(noise)
        out_fake = netD(fake_imgs.detach())
        lossD_fake = criterion(out_fake, fake_label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---- Train Generator ----
        netG.zero_grad()

        out_fake_G = netD(fake_imgs)
        lossG = criterion(out_fake_G, real_label)  # G wants fakes to look real
        lossG.backward()
        optimizerG.step()

    print(f"Epoch {epoch}/{NUM_EPOCHS} | LossD: {lossD.item():.4f} | LossG: {lossG.item():.4f}")

print("Training Completed on Fashion-MNIST!")


# ============================================================
# Final Comparison: Real Fashion vs Generated Fashion Art
# ============================================================

real_imgs, _ = next(iter(dataloader))
real_imgs = real_imgs[:16].cpu()
real_imgs = (real_imgs + 1) / 2  # [-1,1] → [0,1] for viewing

with torch.no_grad():
    noise = torch.randn(16, NZ, 1, 1, device=device)
    fake_imgs = netG(noise).cpu()
    fake_imgs = (fake_imgs + 1) / 2  # [-1,1] → [0,1]

# Stack: first 16 real, next 16 fake
comparison_grid = torch.cat([real_imgs, fake_imgs], dim=0)
grid = make_grid(comparison_grid, nrow=8, padding=2)

os.makedirs("outputs", exist_ok=True)
out_path = "outputs/fashion_real_vs_generated.png"
save_image(grid, out_path)
print("Saved comparison as:", out_path)

# Show inside notebook
from IPython.display import Image, display
display(Image(out_path))

