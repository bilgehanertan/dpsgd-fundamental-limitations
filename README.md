## Fundamental Limitations of Favorable Privacy–Utility Guarantees for DP-SGD

This repo contains:
- **Experiment runs** via `run_dp.py` (writes per-run folders with `config.json`, `summary.csv`, `utility_sigma_th.csv`, …).
- **Reproduction/replay** via `reproduce.py` + the convenience wrappers in `reproduce_scripts/` (replays existing run folders using the saved hyperparameters/seeds and writes `reproduce/utility_sigma_th_reproduced.csv` under each run).
- **The DP-SGD framework** from the JAX Privacy library at version 1.0.0 (https://github.com/google-deepmind/jax_privacy/releases/tag/v1.0.0). To support microbatching, we have utilized the microbatching.py file from the main branch (https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/experimental/microbatching.py).

### Setup
We use Python 3.11.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### Reproduce an existing track (recommended)
Run (Batch Size defaults to 256):

```bash
./reproduce_scripts/reproduce_cifar10_resnet18.sh 256
./reproduce_scripts/reproduce_cifar10_resnet18.sh 512 --only-epochs 1 10
```

#### Tracks (wrappers)
- **CIFAR-10 / ResNet-18**: `./reproduce_scripts/reproduce_cifar10_resnet18.sh <BS>`
- **CIFAR-10 / ViT-Tiny**: `./reproduce_scripts/reproduce_cifar10_vit_tiny_cifar.sh <BS>`
- **CIFAR-10 / ViT-Small**: `./reproduce_scripts/reproduce_cifar10_vit_small_cifar.sh <BS>`
- **CIFAR-10 / ViT-Base**: `./reproduce_scripts/reproduce_cifar10_vit_base_cifar.sh <BS>`
- **CIFAR-100 / ResNet-34**: `./reproduce_scripts/reproduce_cifar100_resnet34.sh <BS>`
- **SVHN / WideResNet28x10**: `./reproduce_scripts/reproduce_svhn_wideresnet28x10.sh <BS>`
- **AGNews / Tx-Small (tiny_128)**: `./reproduce_scripts/reproduce_agnews_tx_small_tiny_128.sh <BS>`
- **AGNews / Tx-Small (small_128)**: `./reproduce_scripts/reproduce_agnews_tx_small_small_128.sh <BS>`
- **AGNews / Tx-Small (bert_base_256)**: `./reproduce_scripts/reproduce_agnews_tx_small_bert_base_256.sh <BS>`

### Run a fresh experiment track (writes new run folders)

Pick a **track name** and write runs into `results/<track>_bs<BS>` so the reproduce wrappers can find them easily.
Example (run both samplers):

```bash
BS=256
TRACK=cifar10_resnet18
python3 run_dp.py --dataset cifar10 --model resnet18 --bs ${BS} --mini_bs 32 --sampler reshuffle --out_dir results/${TRACK}_bs${BS}
python3 run_dp.py --dataset cifar10 --model resnet18 --bs ${BS} --mini_bs 32 --sampler poisson   --out_dir results/${TRACK}_bs${BS}
```

For AGNews, select the text preset:

```bash
BS=256
TRACK=agnews_tx_small_tiny_128
python3 run_dp.py --dataset agnews --model tx_small --tx_preset tiny_128 --bs ${BS} --mini_bs 32 --sampler reshuffle --out_dir results/${TRACK}_bs${BS}
```

### Batch sizes

All tracks can be run for multiple logical batch sizes as long as **`BS % 32 == 0`** (because `mini_bs=32`).
Examples: `128`, `256`, `512`, `1024`, `2048`, …

### Repository layout (file-by-file)

- **`run_dp.py`**: Main experiment runner (JAX). Creates a run directory, writes configs/seeds/summaries, and produces `utility_sigma_th.csv` across multiple epochs/modes.
- **`reproduce.py`**: Reproduce script. Reads an existing run directory, re-runs the existing experiment using the saved hyperparameters/seeds, and writes `reproduce/utility_sigma_th_reproduced.csv`.
- **`reproduce_scripts/`**:  Wrappers to reproduce each track for a given batch size. They search under `results/<track>_bs<BS>/`.
- **`vit_models.py`**: ViT model definitions used by `run_dp.py`.
- **`models_text.py`**: Encoder-only transformer model definitions (Tx-Small variants) used by `run_dp.py`.
- **`batch_selection.py`**: Batch selection utilities used by the experimentsw. This is a modified version of the batch_selection.py file from the JAX Privacy library (https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/batch_selection.py). We have modified slightly to support fixed-size random shuffling and Poisson sampling.
- **`microbatching.py`**: Microbatching utilities (supports `mini_bs` logic). This is the exact same file as in the JAX Privacy library (https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/experimental/microbatching.py).
- **`common.py`**: Shared helpers.
- **`hyperparameter_sweep.py`**: Hyperparameter sweep file for our experiments. This is included for completeness, all hyperparameters used in our experiments can be found in each run directory.
- **`appendix_f.py`**: Appendix F: Figures 1 and 2.

### Results folders and artifacts (what each file means)

Runs are organized as:

- `results/<track>_bs<BS>/`
  - `<dataset>_<model>_<sampler>_bs<BS>_mini_bs32_seed<SEED>_hparam_modeper_epoch/`
    - `config.json`
    - `seeds.json`
    - `summary.csv`
    - `utility_sigma_th.csv`
    - `C_select_clip_only_1epoch.csv`
    - `reproduce/`
      - `utility_sigma_th_reproduced.csv`
      - `reproduce_report.json`

**File meanings**

- **`config.json`**: Full argument snapshot used to create the run folder (dataset/model/sampler/bs/mini_bs, epoch list, sigma mults, tuning templates, etc.).
- **`seeds.json`**: Deterministic seed plan for reproduction. Includes `init_seed` (model init) and per-condition seeds keyed by `(mode, epochs, sigma_mult)`, so epoch-1 / epoch-10 / epoch-25 are reproducible.
- **`summary.csv`**: Small key-value summary (e.g., `M_steps`, selected `C_dagger`, `sigma_th`, etc.).
- **`utility_sigma_th.csv`**: Main results table that is used to populate the tables in the paper. Each row corresponds to a specific `(epochs)` and contains the hyperparameters actually used for that row (`lr`, `momentum`, `wd`) plus DP/clip params (`C_dagger`, `sigma_th`, `sigma_used`) and final metrics (`test_acc`, `test_loss`).
- **`C_select_clip_only_1epoch.csv`**: Intermediate artifact from selecting/validating the clipping constant (clip-only at 1 epoch).
- **`reproduce/utility_sigma_th_reproduced.csv`**: Output of `reproduce.py` for that run folder (same schema as `utility_sigma_th.csv`, recomputed by reproduction).
