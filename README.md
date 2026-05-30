# Assignment 4.1: MNIST Recognition by NN
Train a shallow Multi-Layer Perceptron (MLP), Convolutional Neural Network (CNN), and Transformer (Encoder) on [MNIST dataset](https://huggingface.co/datasets/ylecun/mnist), split into 60k train and 10k test.
Model is trained on training dataset and tested on test dataset.

## Usage
```
pip install -r requirements.txt
python train.py
```

## Takeaway
For this relative small MNIST image datset, CNN captures spatial patterns efficiently, so it performs well on this classification task. While MLP and Transformer help us understand trade-offs between simplicity, spatial modeling, and attention.
