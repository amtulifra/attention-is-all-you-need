# Transformer: Attention Is All You Need

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A clean PyTorch implementation of the Transformer model from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). This project includes training, evaluation, and inference scripts for machine translation tasks.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- NLTK
- tqdm

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/attention-is-all-you-need.git
   cd attention-is-all-you-need
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix/macOS:
   # source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Download Dataset

Download and prepare the Multi30k dataset:
```bash
python download_multi30k.py
```

### Training

Train the model with default settings:
```bash
python train.py --config config/train_config.yaml
```

### Evaluation

Evaluate the trained model:
```bash
python evaluate_final.py --model checkpoints/model_best.pt --data-dir data/multi30k --batch-size 32
```

### Inference

Translate a single sentence:
```bash
python inference.py --model checkpoints/model_best.pt --input "Your input text here"
```

## Project Structure

```
.
├── checkpoints/         # Saved model checkpoints
├── config/              # Configuration files
│   ├── train_config.yaml
│   ├── eval_config.yaml
│   └── inference_config.yaml
├── data/                # Dataset (created after download)
├── src/                 # Source code
│   ├── data/            # Data loading and processing
│   ├── models/          # Model architecture
│   └── utils/           # Utility functions
├── download_multi30k.py # Dataset download script
├── evaluate_final.py    # Evaluation script
├── inference.py         # Inference script
├── requirements.txt     # Dependencies
└── train.py            # Training script
```

## 🚀 Features

- **Complete Transformer Architecture**: Implements both encoder and decoder with multi-head attention
- **Efficient Training**: Supports mixed-precision training and gradient accumulation
- **Flexible Configuration**: Easy to modify model architecture and training parameters
- **Comprehensive Evaluation**: Includes BLEU score calculation, exact match, and token-level accuracy

## 📊 Model Performance

Current metrics on the Multi30k test set:

| Metric | Score | Description |
|--------|-------|-------------|
| BLEU | 0.38 | Translation quality (0-100 scale) |
| Exact Match | 0.0% | Perfect translations |
| Token Accuracy | 17.0% | Correct word choices |
| Length Difference | +12.5 | Predictions are too long |

### Model Architecture

- **Encoder Layers**: 6
- **Decoder Layers**: 6
- **Model Dimension**: 512
- **Feed-Forward Dimension**: 2048
- **Attention Heads**: 8
- **Source/Target Vocabulary**: 8,000 tokens each

### Common Issues

If you encounter issues:
- **Out of memory?** Reduce the batch size (e.g., `--batch-size 16`)
- **File not found?** Verify your file paths and run the download script
- **Shape errors?** Ensure model architecture matches the checkpoint
- **Strange outputs?** Check if the tokenizer matches the training data

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/attention-is-all-you-need.git
   cd attention-is-all-you-need
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n venv python=3.8
   conda activate venv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Quick Start

### 1. Download Dataset

```bash
python download_multi30k.py
```

### 2. Train the Model

```bash
python train.py --config config/train_config.yaml
```

### 3. Evaluate the Model

```bash
python evaluate_final.py --model checkpoints/model_best.pt --data-dir data/multi30k --batch-size 32
```

### 4. Run Inference

```bash
python inference.py --model checkpoints/model_best.pt --input "Your input text here"
```

## 📊 Model Architecture

The Transformer model consists of:

- **Encoder**: Stack of N identical layers with:
  - Multi-head self-attention
  - Position-wise feed-forward network
  - Residual connections and layer normalization

- **Decoder**: Stack of N identical layers with:
  - Masked multi-head self-attention
  - Multi-head attention over encoder output
  - Position-wise feed-forward network
  - Residual connections and layer normalization

## 📈 Training

### Supported Features

- Mixed Precision Training (FP16/FP32)
- Gradient Accumulation
- Learning Rate Scheduling
- Early Stopping
- Model Checkpointing
- TensorBoard Logging

### Training Configuration

Edit `config/train_config.yaml` to customize training parameters:

```yaml
data:
  train_path: data/raw/train.txt
  valid_path: data/raw/valid.txt
  batch_size: 32
  max_length: 100

model:
  d_model: 512
  num_heads: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
  d_ff: 2048
  dropout: 0.1

training:
  epochs: 100
  learning_rate: 0.0001
  warmup_steps: 4000
  label_smoothing: 0.1
  clip_grad_norm: 1.0
  save_dir: checkpoints
  log_dir: runs
```

## 🤖 Inference

### Command Line

```bash
python src/inference.py \
  --model checkpoints/model_best.pt \
  --input "Your input text here" \
  --max_length 100 \
  --beam_size 5
```

## 📊 Evaluation

### BLEU Score

```bash
python src/evaluate.py \
  --model checkpoints/model_best.pt \
  --test_data data/raw/test.txt \
  --output_dir results
```

### Attention Visualization

```python
from src.utils.visualization import plot_attention

# Visualize attention weights
plot_attention(
    attention_weights=attention_weights,
    source_tokens=source_tokens,
    target_tokens=target_tokens,
    layer=0,
    head=0,
    save_path='attention.png'
)
```

### Testing

```bash
pytest tests/
```


## 📝 Notes

- The current model shows signs of overfitting and requires further tuning
- Training logs are saved to `logs/transformer.log`
- Evaluation results are saved to `results/evaluation_results.json`
- Model checkpoints are saved to `checkpoints/`

## 🚧 TODO

### High Priority
- [ ] Implement proper learning rate scheduling with warmup (as per paper: 4000 warmup steps)
- [ ] Add beam search for better inference quality
- [ ] Fix tokenization issues causing poor BLEU scores
- [ ] Implement label smoothing (ε=0.1)
- [ ] Add gradient clipping (paper uses 1.0)

### Model Improvements
- [ ] Weight tying between embedding and output layers
- [ ] Add model checkpoint averaging
- [ ] Implement mixed-precision training
- [ ] Add proper model initialization (paper uses xavier_uniform_)

### Evaluation
- [ ] Add more robust evaluation metrics (e.g., ROUGE, METEOR)
- [ ] Implement proper BLEU tokenization (mteval-v13a.pl)
- [ ] Add validation during training
- [ ] Log attention weights for visualization

### Code Quality
- [ ] Add comprehensive unit tests
- [ ] Improve error handling and logging
- [ ] Add type hints throughout the codebase
- [ ] Document the training process and hyperparameters

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 🤝 Contributing

Contributions are welcome! Please read the general contributing guidelines before submitting pull requests.

- Python 3.7+
- PyTorch 1.9.0+
- torchtext 0.10.0+
- Spacy (for tokenization)
- tqdm (for progress bars)


## Results

On the Multi30k test set, the model currently shows these metrics:

- **BLEU Score**: 0.38 (needs improvement)
- **Exact Match**: 0.0%
- **Token Accuracy**: 17.0%
- **Average Length Difference**: +12.5 tokens

> Note: These results suggest the model requires further training and tuning. The outputs show signs of overfitting or training issues, with repetitive patterns and poor translation quality.

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
2. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [PyTorch Tutorial: Sequence to Sequence with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/translation_transformer.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 attention-is-all-you-need
