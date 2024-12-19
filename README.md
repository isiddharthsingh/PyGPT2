# PyGPT2

This repository contains my implementation of GPT-2 from scratch, inspired by the OpenAI papers on GPT-2 and GPT-3. Unlike the original OpenAI implementation in TensorFlow, this project builds the GPT-2 architecture entirely in PyTorch, enabling a deeper understanding of the model architecture and providing a flexible framework for experimentation.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Evaluating on HellaSwag](#evaluating-on-hellaswag)
   - [Fine-Tuning](#fine-tuning)
6. [Example Outputs](#example-outputs)
7. [Performance Benchmarks](#performance-benchmarks)
8. [References](#references)

## Overview

This project was built to understand and implement GPT-2 from the ground up, based on the GPT-2 and GPT-3 papers. The key accomplishments include:

- **Model Design**: Implemented the GPT-2 model architecture using PyTorch, including the transformer blocks, causal self-attention, and position embeddings.
- **Custom Dataset Processing**: Preprocessed and tokenized the FineWeb-Edu dataset to create a pretraining pipeline.
- **Scalable Training**: Enabled efficient training on GPUs using PyTorch Distributed Data Parallel (DDP).
- **Evaluation**: Evaluated the model's reasoning capabilities on HellaSwag, demonstrating its zero-shot performance.
- **Text Generation**: Trained a 124M parameter GPT-2 model capable of generating coherent and contextually relevant text.

This project is ideal for understanding the internals of GPT-2, customizing its functionality, or using it as a foundation for building advanced natural language processing models.

## Features

- **Custom GPT-2 Implementation**: Designed from scratch in PyTorch with no reliance on pre-existing GPT-2 implementations.
- **Flexible Training Pipeline**: Easily train models on custom datasets.
- **Zero-Shot Evaluation**: Scripts to evaluate reasoning tasks like HellaSwag.
- **Efficient Multi-GPU Support**: Distributed training with PyTorch's DDP framework.
- **Extensible Codebase**: Simplifies model modifications and fine-tuning for various applications.

## Project Structure

```
├── README.md                 # Project documentation
├── fineweb.py                # Dataset processing for pretraining
├── hellaswag.py              # Evaluation on HellaSwag dataset
├── input.txt                 # Sample dataset for testing
├── play.ipynb                # Notebook with training and inference examples
├── train_gpt2_3.py           # Main training script
├── train_gpt2.py             # Small built model

```

## Setup and Installation

1. Clone the repository:

```
git clone <repository-url>
cd <repository-folder>
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. (Optional) Install a GPU-compatible version of PyTorch:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

4. Download datasets for training and evaluation:
   - For FineWeb:

```
python fineweb.py
```

   - For HellaSwag:

```
python hellaswag.py --download
```

## Usage

### Training the Model

To train the GPT-2 model on a dataset:

```
python train_gpt2.py
```

For distributed training with multiple GPUs:

```
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

### Evaluating on HellaSwag

Run evaluation to measure zero-shot performance:

```
python hellaswag.py --model gpt2 --device cuda
```

### Fine-Tuning

To fine-tune the model on a new dataset, replace the training dataset with your custom corpus in the data processing pipeline.

## Example Outputs

After training on 10B tokens, the model generates coherent text. For example:

Input Prompt:
```
Hello, I'm a language model,
```

Generated Output:
```
Hello, I'm a language model, trained to predict the next words based on the text you provide. I can answer questions, write essays, or generate stories.
```

## Performance Benchmarks

- Dataset: FineWeb-Edu
- Model: GPT-2 (124M parameters)
- Training Time: ~1 hour on an NVIDIA V100 GPU
- HellaSwag Accuracy (Zero-Shot):
  - GPT-2 (124M): 29.55% (completion-style)
  - GPT-2-xl: 48.93% (completion-style)

## References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3 Paper](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- [HellaSwag Dataset](https://huggingface.co/datasets/Rowan/hellaswag)
