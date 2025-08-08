# 🫐 Train Your Own Small Language Model

A minimal toolkit for training and using small language models with the Muon optimizer.

## 🚀 Quick Start

### Option 1: Google Colab (No Setup Required)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m9wXIkMlSVW3whSHZiOywMdUClj0amnZ?usp=sharing)

Click the badge above to run everything in your browser with free GPU access!

### Option 2: Local Setup
```bash
# Clone and setup
git clone https://github.com/vukrosic/build-and-release-your-own-llm
cd build-and-release-your-own-llm
python setup.py  # Installs requirements and creates .env file
```

## 🎯 Three Ways to Use This Project

### 1. 🚀 Quick Start - Use My Pre-trained Model

Want to try text generation immediately?

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference with my pre-trained model
python inference.py
```

The script will:
- Show available checkpoints from `vukrosic/blueberry-1`
- Download the model automatically
- Let you generate text interactively

**No setup required!** The model downloads automatically.

### 2. 🏗️ Train Your Own Model

Want to train from scratch?

```bash
# Install dependencies
pip install -r requirements.txt

# Start training (takes ~20 minutes on GPU)
python train_llm.py

# Use your trained model
python inference.py
```

Your model will be saved in `checkpoints/` and you can resume training anytime.

### 3. 📤 Train and Share Your Model

Want to share your model on Hugging Face?

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env file:
# HF_REPO_NAME=your-username/your-model-name
# HF_TOKEN=hf_your_token_here
# PUSH_TO_HUB=true

# 3. Train (uploads automatically)
python train_llm.py
```

Get your HF token from: https://huggingface.co/settings/tokens

## 📁 Project Structure

```
├── train_llm.py       # Training script with Muon optimizer
├── inference.py       # Text generation and model loading
├── setup.py          # Easy setup script
├── requirements.txt   # Python dependencies
├── .env.example      # Environment variables template
└── README.md         # This file
```

## 🎯 What You Get

- **21M parameter transformer model** (384d, 6 layers, 8 heads)
- **Muon optimizer** for efficient training
- **Automatic checkpointing** every 1000 steps
- **Resume training** from any checkpoint
- **Interactive text generation**
- **Hugging Face integration** (optional)

## 📊 Expected Results

- **Training time**: ~16-20 minutes on modern GPU
- **Final perplexity**: ~1.06
- **Model size**: ~21M parameters
- **Memory usage**: ~4-6GB GPU

## 🔧 Customization

### Change Model Size
Edit `train_llm.py`:
```python
@dataclass
class ModelConfig:
    d_model: int = 512      # Bigger model (was 384)
    n_layers: int = 8       # More layers (was 6)
    max_steps: int = 5000  # Train longer for better results (20000)
```

### Use Your Own Data
Edit the dataset loading in `train_llm.py`:
```python
# Replace this line:
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

# With your dataset:
dataset = load_dataset("your-dataset-name", split="train", streaming=True)
```

### Adjust Training Speed
```python
batch_size: int = 16        # Smaller = less memory
gradient_accumulation_steps: int = 8  # Larger = same effective batch size
```

## 📊 Understanding the Output

### During Training
```
Training: 67%|██████▋   | 20000/30000 [12:34<06:15, 26.6it/s, loss=1.234, acc=0.876, ppl=3.4, lr=8.5e-03]
```

- **loss**: Lower is better (target: ~1.0)
- **acc**: Accuracy (target: ~98%)
- **ppl**: Perplexity (target: ~1.1)
- **lr**: Learning rate (automatically scheduled)

### During Inference
```
Prompt: The future of AI is
Generated text: The future of AI is bright and full of possibilities. Machine learning algorithms continue to evolve...
```

## 🚨 Common Issues

### "CUDA out of memory"
```python
# In train_llm.py, reduce batch size:
batch_size: int = 12  # or even 8
```

### "No checkpoints found"
Make sure you've run training first:
```bash
python train_llm.py  # Wait for it to complete
python inference.py  # Now this will work
```

### "HF upload failed"
Check your token permissions:
1. Go to https://huggingface.co/settings/tokens
2. Make sure token has "Write" permission
3. Update your `.env` file

## 🎉 What's Next?

1. **Experiment with prompts** - Try different starting texts
2. **Adjust generation parameters** - Change temperature and top_k in inference.py
3. **Train on your data** - Replace the dataset with your own text
4. **Scale up** - Increase model size for better performance
5. **Share your model** - Upload to Hugging Face for others to use

## 💡 Pro Tips

- **Resume training**: The script automatically detects checkpoints
- **Monitor GPU usage**: Use `nvidia-smi` to check memory usage
- **Save compute**: Use smaller models for experimentation
- **Better results**: More training steps = better model (usually)

Happy training! 🚀