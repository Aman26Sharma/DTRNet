<!--
README for DTRNet & UltraDeep Implementation

This repository implements **DTRNet** (Dynamic Token Routing Network) and its
UltraDeep extensions. DTRNet introduces dynamic token routing to reduce the
quadratic compute cost of self‑attention while updating every token via a
lightweight linear path. UltraDeep explores deeper models, skip‑layer routing,
enhanced routers and normalization schemes. This document provides a
self‑contained overview, setup instructions and guidelines for running
experiments.
-->

# DTRNet & UltraDeep

Dynamic Token Routing for Efficient Transformers

## Overview

**DTRNet** is a Transformer architecture that routes only the most
important tokens through the full self‑attention path and sends the
remaining tokens through a linear projection path. Each token
still receives an explicit update via a shared MLP module.  This
design reduces the quadratic attention cost for most tokens while
maintaining accuracy【791906338155852†screenshot】.  A learned two‑layer router selects
the computation path for each token and the model is trained with
a regularization loss that encourages sparse attention usage【885795514097768†screenshot】.

The **UltraDeep** variants extend DTRNet to very deep settings (up to 92
layers) and introduce nested skip‑layer routing, sequence‑aware routers
and advanced normalization schemes. These modifications stabilize deep
training and allow flexible depth per token.

## Why DTRNet?

Self‑attention has quadratic complexity in the sequence length. DTRNet
cuts this cost by using a *dynamic* routing mechanism: only a small
fraction of tokens follow the attention path while the rest use a
linear update.  This dynamic sparsity is learned rather than fixed and
adaptively adjusts to token importance.  When evaluated on a suite of
language understanding benchmarks, DTRNet achieves a higher average
accuracy than dense baselines (SmolLM and MoD) while using only about
84 % of the FLOPs【505430880371115†screenshot】.  At larger scales (1.3B parameters) it
matches or surpasses baselines on perplexity and accuracy, confirming
that routing tokens can reduce compute without sacrificing performance
【505430880371115†screenshot】.

## Repository Structure

```
.
├── configs/              # training and architecture configs (DTRNet, UltraDeep)
├── models/               # model definitions: DTRNet layers, routers, norms
├── data/                 # dataset loading scripts
├── train.py              # main training entry point
├── evaluate.py           # evaluation scripts (LM harness, long‑context tests)
├── utils/                # utilities: FLOPs counting, routing analysis
├── scripts/              # experiment automation
├── results/              # checkpoints, logs, plots
├── DTRNET_AAAI26.pdf     # anonymized AAAI‑26 submission
└── README.md             # this file
```

## Setup

The recommended environment uses Python 3.10 and PyTorch 2.2.0.
We suggest creating a dedicated conda environment:

```bash
conda create -n dtrnet python=3.10
conda activate dtrnet

# Install PyTorch (select the appropriate CUDA wheel for your system)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install repository dependencies
pip install -r requirements.txt

# Install FlashAttention 2 for variable‑length attention
pip install flash-attn --no-build-isolation
```

### Dataset preparation

The paper evaluates on the **FineWeb‑Edu** dataset and standard language
understanding benchmarks such as ARC, HellaSwag, PIQA and LAMBADA.
For training, prepare your dataset (e.g. FineWeb‑Edu) and set
`--data_path` to its location.  The 360M models are trained on 158
tens of tokens from the FineWeb‑Edu corpus【1844704044110†screenshot】.  The larger 1.3B
models use 10B tokens sampled from the same dataset【1844704044110†screenshot】.  The
tokenizer is LLaMA2 with a vocabulary of 32 000【1844704044110†screenshot】.

## Training

Training is controlled via JSON config files under `configs/`. The key
hyperparameters include the number of layers, hidden size, number of
heads, router type, sparsity regularization strength and batch size.

Example: Train a 360 M‑parameter DTRNet with bilayer routing:

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/dtrnet_bilayer_360m.json \
  --data_path /path/to/fineweb-edu \
  --seq_len 2048 \
  --global_batch_size 384 \
  --save_dir ./checkpoints/dtrnet_bi_360m
```

For UltraDeep experiments, you can enable nested skip‑layer routing and
advanced routers via the config file.  For example, to train a skip‑layer
DTRNet with a Mamba router and HybridNorm:

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/ultradeep_skiplayer.json \
  --norm_method hybridnorm \
  --router_type mamba \
  --aux_loss_coef 0.001
```

Training uses the AdamW optimizer with a peak learning rate of
3e‑4 and a cosine decay schedule【1844704044110†screenshot】.  Gradient clipping is
applied at 1.0, and weight decay is 0.1【1844704044110†screenshot】.  All models use
sequence lengths of 2048 tokens and a global batch size of 384【1844704044110†screenshot】.

## Evaluation and Experiments

DTRNet is evaluated using the
[lm‑evaluation‑harness](https://github.com/EleutherAI/lm-evaluation-harness) for
standard language understanding tasks and a separate script for long
context extrapolation.  After training, run:

```bash
python evaluate.py \
  --model_checkpoint ./checkpoints/dtrnet_bi_360m \
  --tasks arc_challenge,arc_easy,boolq,piqa,hellaswag,tiny_mmlu,winogrande,lambada
```

To assess long‑context performance, enable the `--long_context` flag and
set a YaRN factor (e.g., 10.0) to evaluate sequences up to 20k tokens.
See the `evaluate.py` script for details.

## Results Summary

On the standard LM harness tasks at 360M scale, **DTRNet Bilayer** achieves
an average accuracy of 44.36 %, surpassing SmolLM (44.23 %) and MoD
(43.13 %) while using only 0.84× the FLOPs of the dense baseline
【505430880371115†screenshot】.  This demonstrates that dynamic routing can reduce
compute without sacrificing accuracy.  The Trilayer variant provides
slightly lower average accuracy (43.56 %) but still outperforms some
baselines at reduced cost【505430880371115†screenshot】.  At 1.3B scale, DTRNet
matches or surpasses baselines on perplexity and accuracy while
achieving significant FLOPs savings【505430880371115†screenshot】.

## Citation

If you use this code or build upon it, please cite the anonymous AAAI‐26
submission:

```bibtex
@inproceedings{dtrnet2026,
  title     = {DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers},
  author    = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```
