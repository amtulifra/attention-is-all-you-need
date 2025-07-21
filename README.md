# Transformer: Attention Is All You Need

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A PyTorch implementation of the Transformer model from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). This implementation is designed to be clean, modular, and easy to understand while maintaining high performance.

## 🚀 Features

- **Complete Transformer Architecture**: Implements both encoder and decoder with multi-head attention
- **Efficient Training**: Supports mixed-precision training, gradient accumulation, and multi-GPU training
- **Flexible Configuration**: Easy to modify model architecture and training parameters
- **Comprehensive Evaluation**: Includes BLEU score calculation and attention visualization
- **Production Ready**: Export models to ONNX/TorchScript for deployment
- **Interactive Demo**: Web interface for model inference

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

### Training

```bash
python src/train.py --config config/train_config.yaml
```

### Inference

```bash
python src/inference.py --model checkpoints/model_best.pt --input "Hello, how are you?"
```

### Web Demo

```bash
python src/app.py
```

## 🏗️ Project Structure

```
.
├── config/                 # Configuration files
│   ├── train_config.yaml   # Training configuration
│   └── model_config.yaml   # Model architecture configuration
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model definitions
│   ├── utils/              # Utility functions
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Inference script
│   └── app.py              # Web interface
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── checkpoints/            # Saved models
├── runs/                   # TensorBoard logs
├── requirements.txt        # Python dependencies
└── README.md              # This file
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

### Python API

```python
from src.models.transformer import Transformer
from src.utils.tokenizer import Tokenizer

# Load model
model = Transformer.load_from_checkpoint('checkpoints/model_best.pt')
model.eval()

# Tokenize input
input_ids = tokenizer.encode("Your input text here")

# Generate output
output_ids = model.generate(input_ids, max_length=100)
output_text = tokenizer.decode(output_ids)
print(output_text)
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

## 🛠 Development

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run the following commands before committing:

```bash
black .
isort .
flake8
mypy .
```

### Testing

```bash
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📧 Contact

For questions or feedback, please open an issue or contact [Your Name] at [your.email@example.com].

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

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