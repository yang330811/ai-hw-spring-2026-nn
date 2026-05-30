# Assignment 4.1: MNIST Recognition by NN
Train a shallow Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Transformer (Encoder) on [MNIST dataset](https://huggingface.co/datasets/ylecun/mnist), split into 60k train and 10k test.
Model is trained on training dataset and tested on test dataset.

## Usage
```
pip install -r requirements.txt
python train.py
```

## Dataset

- **60k train / 10k test** (official MNIST split via `torchvision.datasets.MNIST`)
- **28×28 grayscale** digits, **10 classes** (0–9)
- Preprocessing: `ToTensor()` only — pixels scaled to **[0, 1]**
- `DataLoader`: batch size **128**; `shuffle=True` for train, `shuffle=False` for test
- Train on the training set only; evaluate on the held-out test set

## Models

**MLP** — Flatten 784 pixels → Linear(256) → ReLU → Linear(10). Simple baseline; ignores spatial structure.

**CNN** — Two conv blocks (Conv → ReLU → MaxPool): `1×28×28 → 32×14×14 → 64×7×7`, then flatten → Linear(128) → ReLU → Linear(10). Learns local image patterns.

**Transformer** — Split image into **16 patches** (7×7 each), project with a linear layer, add [CLS] token and positional embeddings, pass through 2 encoder layers (4 attention heads), classify from CLS output.

## Training

- Optimizer: **Adam**, lr **1e-3**
- Loss: **CrossEntropyLoss**
- Batch size: **128**, epochs: **5**
- Standard loop: `train()` → forward → loss → backward → `step()`; evaluate each epoch with `eval()` + `no_grad()`

## Results (5 epochs, CPU)

| Model | Final Test Accuracy | Parameters |
|-------|--------------------:|-----------:|
| CNN | 98.90% | ~422K |
| MLP | 97.65% | ~204K |
| Transformer | 96.89% | ~72K |

Full per-epoch history saved in `results/all_results.json`.

## Takeaway

For this relatively small MNIST image dataset, **CNN** captures spatial patterns efficiently and performs best. **MLP** and **Transformer** help illustrate trade-offs between simplicity, spatial modeling, and attention — architecture matters even on the same data.
