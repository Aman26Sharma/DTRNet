# DTRNet

Dynamic Token-Routed Network (DTRNet) — a routed Transformer variant that adapts per-token compute by selectively
skipping or executing attention/MLP sublayers. This repository provides a minimal, **self-contained** scaffold with
correct paths so you can plug in your modeling code immediately.

> [Unverified] The claims below reflect our internal observations (May–June 2025). They may not generalize to all settings.

## Why DTRNet?

- Token-choice routing at train and inference time removes the train–test mismatch common in expert-choice routing.
- Optional skip paths allow dynamic depth per token to reduce redundant compute on “easy” tokens.
- Normalization strategies for deep models (e.g., LayerNormScaling, HybridNorm) stabilize >32-layer regimes.
- [Unverified] In our small-scale runs on SmolLM-360M, DTRNet variants achieved comparable average accuracy at lower FLOPs
  than a dense baseline, with the best trade-offs observed between 0.7–0.9× FLOPs.

## Repository Structure

```
DTRNet/
├─ dtrnet/
│  ├─ __init__.py
│  ├─ train.py                     # placeholder entrypoint; replace with your training loop
│  └─ configs/
│     ├─ smolm360m_dtrnet_bilayer.yaml
│     ├─ smolm360m_dtrnet_four_attn.yaml
│     └─ smolm360m_dtrnet_zero_attn.yaml
├─ scripts/
│  ├─ train.sh
│  ├─ eval_lm_harness.sh
│  └─ prepare_data.sh
├─ eval/
│  └─ longbench_config.json
├─ experiments/                   # put your runs, logs, result JSONs here
│  └─ README.md
├─ docs/
│  ├─ README.md
│  └─ UltraDeep.pdf               # included if provided
├─ data/                          # optional local data/cache dir
└─ environment.yml
```

All paths referenced below exist in this scaffold.

## Quickstart

### 1) Create the environment

```bash
conda env create -f environment.yml
conda activate dtrnet
```

> If `flash-attn` fails to install on your system, remove it from `environment.yml`, or install from source following their docs.
> You can still run without flash-attn (slower).

### 2) (Optional) Authenticate with Hugging Face

```bash
huggingface-cli login
```

### 3) Train

Use the included script; it calls the placeholder entrypoint `dtrnet.train` so the path is valid.
Replace it later with your production training code.

```bash
bash scripts/train.sh dtrnet/configs/smolm360m_dtrnet_bilayer.yaml outputs/smolm360m_bilayer_run
```

You should see a placeholder artifact written to `outputs/smolm360m_bilayer_run/` confirming paths are correct.

Alternative configs (same command, different config):

- `dtrnet/configs/smolm360m_dtrnet_four_attn.yaml`
- `dtrnet/configs/smolm360m_dtrnet_zero_attn.yaml`

### 4) Evaluation

#### lm-eval-harness

Edit and run the placeholder script with your checkpoint:

```bash
bash scripts/eval_lm_harness.sh
```

A typical command (adjust paths) would look like:

```bash
lm_eval --model hf   --model_args pretrained=PATH_OR_HUB_ID   --tasks arc_easy,hellaswag,piqa,winogrande,boolq,obqa,tiny_mmlu   --batch_size auto   --output_path outputs/lm_harness_results.json
```

#### LongBench (example)

Put your task selection in `eval/longbench_config.json`, then adapt your LongBench runner to load from there.

## Configuration Notes

The sample YAMLs under `dtrnet/configs/` include options you likely need:

- `dtrnet.router_type`: `token_choice` (recommended) or `expert_choice`.
- `dtrnet.aux_loss_lambda`: coefficient for the load-balancing/usage auxiliary loss.
- `dtrnet.safe_min_attn_ratio`: lower bound to avoid starving attention.
- `dtrnet.normalization`: `lns` (LayerNormScaling), `hybridnorm`, or `none`.
- `dtrnet.architecture`: `bilayer`, `four_attn`, `zero_attn`, or your custom layout.

> Tuning tips (from our internal notes; treat as hypotheses until you verify in your setup):
> - `aux_loss_lambda` in `[1e-5, 1e-3]` with gradual warmup often prevents pathological routing.
> - A small but non-zero `safe_min_attn_ratio` (e.g., 0.05–0.15) avoids degenerate “all-skip” failure modes.
> - If you go deep (>48 layers), try `lns` and/or pre-norm with scaled residuals.

## Data

The placeholder train script streams from HF datasets as an example. For serious runs, prepare a curated corpus
(e.g., FineWeb, Wiki, code) and shard it. Use `scripts/prepare_data.sh` as a starting point.

## Reproducing Common Layouts

- **Bilayer (T, D, T, D, …)**: `dtrnet/configs/smolm360m_dtrnet_bilayer.yaml`
- **4-Attn (Two at start, two in middle; rest routed)**: `dtrnet/configs/smolm360m_dtrnet_four_attn.yaml`
- **0-Attn (all routed/MLP-only blocks)**: `dtrnet/configs/smolm360m_dtrnet_zero_attn.yaml`

> These configs are templates; wire them to your modeling code to enforce the layout you need.

## Logging & Checkpoints

Use your preferred stack (W&B, TensorBoard, JSONL logs). The scaffold writes minimal artifacts into `outputs/`.

## Troubleshooting

- **flash-attn install fails**: remove the line in `environment.yml` and retry; ensure CUDA/driver compatibility.
- **OOM**: reduce `per_device_train_batch_size`, enable gradient checkpointing, or shorten `block_size`.
- **Router collapse**: decrease `aux_loss_lambda`, raise `safe_min_attn_ratio`, or add entropy/jitter to gating.

## License

This scaffold is provided as-is for you to adapt to your internal DTRNet code.

