---
language:
- en
license: mit
pipeline_tag: text-generation
tags:
- pytorch
- transformer
- language-model
- muon-optimizer
- small-model
- llm-training
- educational
datasets:
- HuggingFaceTB/smollm-corpus
metrics:
- perplexity
- accuracy
base_model: HuggingFaceTB/SmolLM-135M
library_name: pytorch
---

# 🫐 Build and Release Your Own LLM

A complete toolkit for training, evaluating, and deploying your own small language model using the efficient Muon optimizer.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Installation
```bash
git clone <your-repo>
cd build-and-release-your-own-llm
pip install -r requirements.txt
```

### Setup Configuration
```bash
python setup_config.py
```
This will prompt you for your Hugging Face repository name and token.

## 📋 Project Structure

```
├── train_llm.py           # Main training script
├── inference.py           # Text generation and model testing
├── generate_model_card.py # Create comprehensive model documentation
├── upload_model_card.py   # Upload model and documentation to HF Hub
├── setup_config.py        # Configuration setup helper
├── test_hf_upload.py      # Test Hugging Face integration
├── requirements.txt       # Python dependencies
├── checkpoints/           # Model checkpoints (created during training)
├── data_cache/           # Cached tokenized data
└── README.md             # This file
```

## 🎯 How to Use

### 1. Training Your Model

**Basic Training:**
```bash
python train_llm.py
```

**Resume from Checkpoint:**
The script will automatically detect existing checkpoints and ask if you want to resume training.

**Key Training Features:**
- **Muon Optimizer**: Advanced momentum-based optimizer with Newton-Schulz orthogonalization
- **Mixed Precision**: Automatic mixed precision training for efficiency
- **Gradient Accumulation**: Effective batch size scaling
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Automatic Checkpointing**: Saves model state every 1000 steps
- **Hugging Face Integration**: Optional automatic upload to HF Hub

**Training Configuration:**
- Model: 21M parameters (384d, 6 layers, 8 heads)
- Dataset: HuggingFaceTB/smollm-corpus (Cosmopedia-v2)
- Training Steps: 30,000 (configurable)
- Batch Size: 24 with 4-step gradient accumulation
- Learning Rate: 0.01 (Muon), 0.001 (AdamW for embeddings)

### 2. Text Generation and Inference

**Interactive Generation:**
```bash
python inference.py
```

This script will:
- Show available local checkpoints
- Check Hugging Face for additional checkpoints
- Allow you to download missing checkpoints
- Load your selected model for interactive text generation

**Features:**
- Automatic checkpoint discovery (local + HF Hub)
- Interactive model selection
- Configurable generation parameters (temperature, top-k)
- GPU acceleration when available

### 3. Generate Model Documentation

**Create Comprehensive Model Card:**
```bash
python generate_model_card.py
```

This generates:
- Training curve visualizations
- Model architecture diagrams
- Performance comparison charts
- Detailed README with metrics and usage instructions

### 4. Upload to Hugging Face Hub

**Upload Model and Documentation:**
```bash
python upload_model_card.py
```

This will:
- Upload your trained model
- Upload generated documentation and plots
- Create a professional model card on HF Hub
- Set up proper model tags and metadata

## 🔧 Configuration Options

### Environment Variables (.env file)
```bash
HF_REPO_NAME=your-username/your-model-name
HF_TOKEN=hf_your_token_here
PUSH_TO_HUB=true
SAVE_EVERY=1000
```

### Model Configuration (in train_llm.py)
```python
@dataclass
class ModelConfig:
    # Architecture
    d_model: int = 384          # Model dimension
    n_heads: int = 8            # Attention heads
    n_layers: int = 6           # Transformer layers
    d_ff: int = 1536           # Feed-forward dimension
    
    # Training
    batch_size: int = 24        # Batch size
    max_steps: int = 30000      # Training steps
    muon_lr: float = 0.01       # Learning rate
    
    # Data
    max_seq_len: int = 512      # Sequence length
    num_documents: int = 2000   # Training documents
    max_tokens: int = 500000    # Training tokens
```

## 📊 Expected Results

With the default configuration, you can expect:
- **Training Time**: ~16-20 minutes on modern GPU
- **Final Perplexity**: ~1.06
- **Validation Accuracy**: ~98.6%
- **Model Size**: ~21M parameters
- **Memory Usage**: ~4-6GB GPU memory

## 🛠️ Advanced Usage

### Custom Dataset
Modify the `load_and_cache_data()` function in `train_llm.py` to use your own dataset:

```python
# Replace this line:
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

# With your dataset:
dataset = load_dataset("your-dataset-name", split="train", streaming=True)
```

### Hyperparameter Tuning
Key parameters to experiment with:
- `muon_lr`: Learning rate (try 0.005-0.02)
- `n_layers`: Model depth (try 4-12)
- `d_model`: Model width (try 256-768)
- `max_steps`: Training duration
- `batch_size`: Memory vs. speed tradeoff

### Multi-GPU Training
The code supports single GPU training. For multi-GPU, consider using:
- `torch.nn.DataParallel`
- `torch.distributed`
- Hugging Face Accelerate

## 🔍 Monitoring Training

### Real-time Metrics
The training script displays:
- Loss curves (training and validation)
- Accuracy progression
- Perplexity trends
- Learning rate schedule
- Gradient norms

### Checkpoints
- Automatic saving every 1000 steps
- Resume capability from any checkpoint
- Hugging Face Hub integration
- Local backup in `checkpoints/` directory

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce `batch_size` (try 16 or 12)
- Reduce `max_seq_len` (try 256)
- Enable gradient checkpointing

**Slow Training:**
- Increase `batch_size` if memory allows
- Use `num_workers=4` in DataLoader
- Ensure CUDA is properly installed

**Poor Convergence:**
- Adjust learning rate (`muon_lr`)
- Increase `max_steps`
- Check data quality and preprocessing

**Hugging Face Upload Issues:**
- Verify your token has write permissions
- Check repository name format
- Ensure stable internet connection

### Debug Mode
Add debug prints by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Learning Resources

### Understanding the Code
- **Muon Optimizer**: Novel momentum-based optimizer with orthogonalization
- **Transformer Architecture**: Decoder-only model with RoPE and RMSNorm
- **Mixed Precision**: Automatic mixed precision for efficiency
- **Gradient Accumulation**: Effective batch size scaling

### Further Reading
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformers library and model hosting
- **SmolLM team** for the high-quality training corpus
- **Muon optimizer** authors for the innovative optimization technique
- **PyTorch team** for the deep learning framework

---

**Happy training! 🚀**
