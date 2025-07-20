# Attention Is All You Need - Transformer Implementation

This repository contains a PyTorch implementation of the Transformer model introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). The implementation includes both the encoder and decoder architecture with multi-head self-attention mechanisms.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [References](#references)

## Overview

The Transformer is a novel neural network architecture that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely. This implementation follows the original paper's architecture and includes:

- Multi-Head Self-Attention
- Position-wise Feed-Forward Networks
- Positional Encoding
- Residual Connections and Layer Normalization
- Masked Multi-Head Attention (for decoder)

## Model Architecture

The Transformer model consists of:

1. **Encoder**: A stack of N=6 identical layers, each with:
   - Multi-head self-attention mechanism
   - Position-wise fully connected feed-forward network
   - Residual connections and layer normalization

2. **Decoder**: A stack of N=6 identical layers, each with:
   - Masked multi-head self-attention
   - Multi-head attention over encoder output
   - Position-wise feed-forward network
   - Residual connections and layer normalization

## Features

- **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces.
- **Positional Encoding**: Injects information about the relative or absolute position of tokens in the sequence.
- **Residual Connections**: Help with the flow of gradients through the network.
- **Layer Normalization**: Normalizes the inputs across the features.
- **Label Smoothing**: Used during training for better generalization.
- **Beam Search**: For better translation quality during inference.

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- torchtext 0.10.0+
- Spacy (for tokenization)
- tqdm (for progress bars)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/attention-is-all-you-need.git
   cd attention-is-all-you-need
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required language models for Spacy:
   ```bash
   python -m spacy download de_core_news_sm
   python -m spacy download en_core_web_sm
   ```

## Training

To train the model on the Multi30k dataset (German to English translation):

```bash
python train.py
```

Training parameters can be modified in the `Config` class in `train.py`.

## Inference

To use the trained model for translation:

```bash
python inference.py --model checkpoints/model.pt --vocab vocab.pt
```

This will start an interactive session where you can input German sentences to be translated to English.

## Results

On the Multi30k test set, the model achieves:

- BLEU Score: ~[Your BLEU score here]
- Training Loss: ~[Your training loss here]
- Validation Loss: ~[Your validation loss here]

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
2. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [PyTorch Tutorial: Sequence to Sequence with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/translation_transformer.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 attention-is-all-you-need