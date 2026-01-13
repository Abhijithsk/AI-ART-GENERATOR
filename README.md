👗 Fashion-MNIST DCGAN
A stable Deep Convolutional GAN that generates novel fashion sketches (grayscale items like shirts, bags, and shoes) trained on the Fashion-MNIST dataset.

⚡ Quick Start
1. Install Requirements
Bash

pip install torch torchvision
2. Run Training
Bash

python main.py
Data: Automatically downloads to data_fashion/.

Training: Runs for 20 epochs (default).

Results: Saves a Real vs. Generated comparison to outputs/.

🧠 Key Features
Architecture: DCGAN (Radford et al.) with Transposed Convolutions.

Stability: Implements Label Smoothing (0.9 for real labels) and BCEWithLogitsLoss (no Sigmoid in Discriminator) to prevent mode collapse.

Output: Upscales 28x28 inputs to 64x64 high-fidelity generations.
