# CPU-Optimized Vision Transformer Implementation

This project demonstrates how to optimize a Vision Transformer (ViT) model using ONNX Runtime and OpenMP for CPU acceleration. The implementation includes training, optimization, and benchmarking capabilities.

## Features
- Vision Transformer (ViT) implementation using PyTorch
- ONNX Runtime optimization for CPU inference
- OpenMP multi-threading support
- Benchmarking tools for performance comparison
- CPU-specific optimizations

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training script:
```bash
python train_vit.py
```

3. Run benchmarking:
```bash
python benchmark.py
```

## Project Structure
- `train_vit.py`: Main training script for ViT model
- `model.py`: ViT model implementation
- `optimizer.py`: ONNX conversion and optimization utilities
- `benchmark.py`: Performance benchmarking tools
- `utils.py`: Helper functions and utilities
