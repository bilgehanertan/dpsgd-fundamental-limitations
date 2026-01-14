# dpv4.py
import argparse
import csv
import json
import os
import math
import hashlib
import random
from typing import Optional, Tuple, List, Dict, Any

from models_text import TX_PRESETS
from vit_models import ViTConfig, ViTClassifier, VIT_PRESETS

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common import (
    ResNet18,
    ResNet34,
    WideResNet28x10,
    load_cifar10_arrays,
    load_cifar100_arrays,
    load_svhn_arrays,
    compute_sigma_th,
    load_agnews_arrays,
)

import batch_selection as subsampling
import microbatching

from jax_privacy.dp_sgd import grad_clipping as dp_grad_clipping
from jax_privacy.dp_sgd import gradients as dp_gradients
from jax_privacy.dp_sgd import typing as dp_types

from models_text import TransformerConfig, TransformerClassifier


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "cifar10": {
        "num_classes": 10,
        "array_loader": load_cifar10_arrays,
        "input_shape": (1, 32, 32, 3),
        "batch_kind": "vision",
    },
    "cifar100": {
        "num_classes": 100,
        "array_loader": load_cifar100_arrays,
        "input_shape": (1, 32, 32, 3),
        "batch_kind": "vision",
    },
    "svhn": {
        "num_classes": 10,
        "array_loader": lambda: load_svhn_arrays(include_extra=True),
        "input_shape": (1, 32, 32, 3),
        "batch_kind": "vision",
    },
    "agnews": {
        "num_classes": 4,
        "array_loader": None,
        "input_shape": None,
        "batch_kind": "text",
    },
}

MODEL_REGISTRY: Dict[str, Any] = {
    "resnet18": lambda num_classes, meta=None: ResNet18(num_classes=num_classes),
    "resnet34": lambda num_classes, meta=None: ResNet34(num_classes=num_classes),
    "wideresnet28x10": lambda num_classes, meta=None: WideResNet28x10(
        num_classes=num_classes
    ),
    "tx_small": lambda num_classes, meta: TransformerClassifier(
        TransformerConfig(
            vocab_size=int(meta["vocab_size"]),
            num_classes=int(num_classes),
            max_len=int(meta["max_len"]),
            d_model=int(meta.get("d_model", 256)),
            n_heads=int(meta.get("n_heads", 4)),
            n_layers=int(meta.get("n_layers", 4)),
            d_ff=int(meta.get("d_ff", 1024)),
            dropout=0.0,
        )
    ),
    "vit_tiny_cifar": lambda num_classes, meta=None: ViTClassifier(
        ViTConfig(
            num_classes=int(num_classes),
            image_size=32,
            dropout=0.0,
            attn_dropout=0.0,
            **VIT_PRESETS["vit_tiny_cifar"],
        )
    ),
    "vit_small_cifar": lambda num_classes, meta=None: ViTClassifier(
        ViTConfig(
            num_classes=int(num_classes),
            image_size=32,
            dropout=0.0,
            attn_dropout=0.0,
            **VIT_PRESETS["vit_small_cifar"],
        )
    ),
    "vit_base_cifar": lambda num_classes, meta=None: ViTClassifier(
        ViTConfig(
            num_classes=int(num_classes),
            image_size=32,
            dropout=0.0,
            attn_dropout=0.0,
            **VIT_PRESETS["vit_base_cifar"],
        )
    ),
}


def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_row(path: str, header: List[str], row: List):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _derive_seed(master_seed: int, *tags: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(master_seed)).encode("utf-8"))
    for t in tags:
        h.update(b"|")
        h.update(str(t).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % (2**31 - 1)


def load_best_hparams_for_epoch(
    template_path: str,
    epoch: int,
    bs: int,
    fallback: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    path = template_path.format(E=epoch)
    if not os.path.exists(path):
        return fallback

    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0].strip() == "dataset":
                continue
            if len(row) < 7:
                continue
            try:
                row_bs = int(row[3])
            except Exception:
                continue
            if row_bs == int(bs):
                try:
                    lr = float(row[4])
                    mom = float(row[5])
                    wd = float(row[6])
                    return lr, mom, wd
                except Exception:
                    return fallback
    return fallback


def forward_logits(model, params, batch: Dict[str, jnp.ndarray], train: bool):
    if "image" in batch:
        return model.apply({"params": params}, batch["image"], train=train)
    # text
    return model.apply({"params": params}, batch["tokens"], batch["attn"], train=train)


def make_batch_fns(batch_kind: str, arrays: Dict[str, Any]):
    if batch_kind == "vision":
        train_images = arrays["train_images"]
        train_labels = arrays["train_labels"]
        test_images = arrays["test_images"]
        test_labels = arrays["test_labels"]

        def make_train_batch(
            safe_idx: np.ndarray, is_padding: np.ndarray
        ) -> Dict[str, jnp.ndarray]:
            return {
                "image": train_images[safe_idx],
                "label": train_labels[safe_idx],
                "is_padding_example": jnp.array(is_padding),
            }

        def make_eval_batch(s: int, e: int) -> Dict[str, jnp.ndarray]:
            return {"image": test_images[s:e], "label": test_labels[s:e]}

        def get_num_test() -> int:
            return int(test_images.shape[0])

        return make_train_batch, make_eval_batch, get_num_test

    if batch_kind == "text":
        train_tokens = arrays["train_tokens"]
        train_attn = arrays["train_attn"]
        train_labels = arrays["train_labels"]
        test_tokens = arrays["test_tokens"]
        test_attn = arrays["test_attn"]
        test_labels = arrays["test_labels"]

        def make_train_batch(
            safe_idx: np.ndarray, is_padding: np.ndarray
        ) -> Dict[str, jnp.ndarray]:
            return {
                "tokens": train_tokens[safe_idx],
                "attn": train_attn[safe_idx],
                "label": train_labels[safe_idx],
                "is_padding_example": jnp.array(is_padding),
            }

        def make_eval_batch(s: int, e: int) -> Dict[str, jnp.ndarray]:
            return {
                "tokens": test_tokens[s:e],
                "attn": test_attn[s:e],
                "label": test_labels[s:e],
            }

        def get_num_test() -> int:
            return int(test_tokens.shape[0])

        return make_train_batch, make_eval_batch, get_num_test

    raise ValueError(f"Unknown batch_kind: {batch_kind}")


def make_eval_step(model):
    @jax.jit
    def eval_step(params, batch):
        logits = forward_logits(model, params, batch, train=False)
        acc = jnp.mean(jnp.argmax(logits, -1) == batch["label"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return {"loss": loss, "acc": acc}

    return eval_step


def eval_full(
    params, eval_step, make_eval_batch, num_test: int, physical_bs: int
) -> Tuple[float, float]:
    num_batches = num_test // physical_bs
    total_acc, total_loss = 0.0, 0.0
    for i in range(num_batches):
        s = i * physical_bs
        e = s + physical_bs
        batch = make_eval_batch(s, e)
        m = eval_step(params, batch)
        total_acc += float(m["acc"])
        total_loss += float(m["loss"])
    return total_acc / max(num_batches, 1), total_loss / max(num_batches, 1)


def make_sampler(sampler_name: str, num_train: int, bs: int, epochs: int):
    if sampler_name == "reshuffle":
        steps_per_epoch = int(num_train // bs)
        sampler = subsampling.FixedBatchShufflingSampling(
            batch_size=int(bs), epochs=int(epochs)
        )
        return sampler, steps_per_epoch
    elif sampler_name == "poisson":
        q = bs / num_train
        steps_per_epoch = int(num_train / bs)
        iterations = epochs * steps_per_epoch
        sampler = subsampling.PoissonSubsampling(sampling_prob=q, iterations=iterations)
        return sampler, steps_per_epoch
    else:
        raise ValueError(f"Unknown sampler {sampler_name}")


# ----------------------------
# Clean SGD training
# ----------------------------
def train_clean_sgd(
    *,
    model,
    params_init,
    epochs: int,
    logical_bs: int,
    physical_bs: int,
    lr: float,
    wd: float,
    momentum: float,
    sampler_name: str,
    seed: int,
    subsample_seed: Optional[int],
    num_train: int,
    make_train_batch,
) -> Tuple[dict, int]:
    tx = optax.chain(
        optax.add_decayed_weights(wd),
        optax.sgd(learning_rate=lr, momentum=momentum),
    )
    opt_state = tx.init(params_init)

    def microbatch_sum_grads_and_loss(params, batch):
        is_padding = batch.get("is_padding_example")
        if is_padding is None:
            is_real = jnp.ones((batch["label"].shape[0],), dtype=jnp.float32)
        else:
            is_real = 1.0 - is_padding.astype(jnp.float32)

        def sum_loss_fn(p):
            logits = forward_logits(model, p, batch, train=True)
            per_example_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["label"]
            )
            return jnp.sum(per_example_loss * is_real)

        sum_loss, sum_grads = jax.value_and_grad(sum_loss_fn)(params)
        real_count = jnp.sum(is_real)
        return sum_grads, sum_loss, real_count

    if int(physical_bs) == int(logical_bs):
        microbatched_sum_fn = microbatch_sum_grads_and_loss
    else:
        microbatched_sum_fn = microbatching.microbatch(
            fun=microbatch_sum_grads_and_loss,
            batch_argnums=1,
            microbatch_size=physical_bs,
            accumulation_type=(
                microbatching.AccumulationType.SUM,
                microbatching.AccumulationType.SUM,
                microbatching.AccumulationType.SUM,
            ),
        )

    @jax.jit
    def update_step(params, opt_state, batch):
        sum_grads, sum_loss, real_bs = microbatched_sum_fn(params, batch)
        real_bs = jnp.maximum(real_bs, 1.0)
        denom_bs = jnp.asarray(
            float(logical_bs) if sampler_name == "poisson" else real_bs,
            dtype=jnp.float32,
        )
        grads = jax.tree_util.tree_map(lambda g: g / denom_bs, sum_grads)
        loss = sum_loss / denom_bs
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    sampler, _ = make_sampler(sampler_name, num_train, logical_bs, epochs=epochs)
    current_subsample_seed = subsample_seed if subsample_seed is not None else seed
    sampler_iter = sampler.batch_iterator(num_train, rng=current_subsample_seed)

    params = params_init
    step = 0
    for batch_indices in sampler_iter:
        step += 1
        padded = subsampling.pad_to_multiple_of(np.asarray(batch_indices), physical_bs)
        is_padding = (padded == -1).astype(np.float32)
        safe = np.where(padded == -1, 0, padded)

        batch = make_train_batch(safe, is_padding)
        params, opt_state, _ = update_step(params, opt_state, batch)

    return params, step


# ----------------------------
# DP-SGD training (clip + noise)
# ----------------------------
def train_full_dpsgd(
    *,
    model,
    params_init,
    epochs: int,
    logical_bs: int,
    physical_bs: int,
    lr: float,
    wd: float,
    momentum: float,
    sampler_name: str,
    seed: int,
    subsample_seed: Optional[int],
    C: float,
    sigma: float,
    num_train: int,
    make_train_batch,
) -> Tuple[dict, int]:
    tx = optax.chain(
        optax.add_decayed_weights(wd),
        optax.sgd(learning_rate=lr, momentum=momentum),
    )
    opt_state = tx.init(params_init)

    gradient_computer = dp_gradients.DpsgdGradientComputer(
        clipping_norm=float(C),
        noise_multiplier=float(sigma),
        rescale_to_unit_norm=False,
        per_example_grad_method=dp_grad_clipping.VECTORIZED,
    )

    def microbatch_sum_clipped_grads_and_real_count(params, rng, batch):
        is_padding = batch.get("is_padding_example")
        if is_padding is None:
            is_real = jnp.ones((batch["label"].shape[0],), dtype=jnp.float32)
        else:
            is_real = 1.0 - is_padding.astype(jnp.float32)

        def loss_fn(p, network_state, rng_per_example, inputs):
            del network_state, rng_per_example
            if "image" in inputs:
                fwd_batch = {"image": inputs["image"]}
            else:
                fwd_batch = {"tokens": inputs["tokens"], "attn": inputs["attn"]}
            logits = forward_logits(model, p, fwd_batch, train=True)
            per_ex = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=inputs["label"]
            )
            loss = jnp.mean(per_ex * inputs["is_real"])

            return loss, ({}, dp_types.Metrics())

        inputs = {"label": batch["label"], "is_real": is_real}
        if "image" in batch:
            inputs["image"] = batch["image"]
        else:
            inputs["tokens"] = batch["tokens"]
            inputs["attn"] = batch["attn"]

        (_, _), avg_clipped_grads = gradient_computer.loss_and_clipped_gradients(
            loss_fn=loss_fn,
            params=params,
            network_state={},
            rng_per_local_microbatch=rng,
            inputs=inputs,
        )

        sum_grads = jax.tree_util.tree_map(
            lambda g: g * float(physical_bs), avg_clipped_grads
        )
        real_count = jnp.sum(is_real)
        return sum_grads, real_count

    if int(physical_bs) == int(logical_bs):
        microbatched_sum_fn = microbatch_sum_clipped_grads_and_real_count
    else:
        microbatched_sum_fn = microbatching.microbatch(
            fun=microbatch_sum_clipped_grads_and_real_count,
            batch_argnums=2,
            microbatch_size=physical_bs,
            accumulation_type=(
                microbatching.AccumulationType.SUM,
                microbatching.AccumulationType.SUM,
            ),
        )

    @jax.jit
    def update_step(params, opt_state, noise_state, rng, batch):
        rng, grad_rng, noise_rng = jax.random.split(rng, 3)
        sum_grads, real_bs = microbatched_sum_fn(params, grad_rng, batch)

        ## Divide by expected:
        real_bs = jnp.maximum(real_bs, 1.0)
        denom_bs = jnp.asarray(
            float(logical_bs) if sampler_name == "poisson" else real_bs,
            dtype=jnp.float32,
        )
        grads = jax.tree_util.tree_map(lambda g: g / denom_bs, sum_grads)

        ## Divide by realized:
        # real_bs = jnp.maximum(real_bs, 1.0)
        # denom_bs = real_bs  # use realized batch size for *all* samplers
        # grads = jax.tree_util.tree_map(lambda g: g / denom_bs, sum_grads)

        noisy_grads, _, new_noise_state = gradient_computer.add_noise_to_grads(
            grads, noise_rng, denom_bs, noise_state
        )
        updates, new_opt_state = tx.update(noisy_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, new_noise_state, rng

    sampler, _ = make_sampler(sampler_name, num_train, logical_bs, epochs=epochs)
    current_subsample_seed = subsample_seed if subsample_seed is not None else seed
    sampler_iter = sampler.batch_iterator(num_train, rng=current_subsample_seed)

    loop_key, noise_key = jax.random.split(jax.random.PRNGKey(seed))
    noise_state = gradient_computer.init_noise_state(noise_key)
    rng = loop_key

    params = params_init
    step = 0
    for batch_indices in sampler_iter:
        step += 1
        padded = subsampling.pad_to_multiple_of(np.asarray(batch_indices), physical_bs)
        is_padding = (padded == -1).astype(np.float32)
        safe = np.where(padded == -1, 0, padded)

        batch = make_train_batch(safe, is_padding)
        params, opt_state, noise_state, rng = update_step(
            params, opt_state, noise_state, rng, batch
        )

    return params, step


# ----------------------------
# Choose C_dagger via 1 epoch clip-only (sigma=lower_bound)
# ----------------------------
def select_C_dagger_one_epoch(
    *,
    model,
    params_init,
    logical_bs,
    physical_bs,
    lr,
    wd,
    momentum,
    sampler_name,
    seed,
    subsample_seed,
    C_grid: List[float],
    out_csv: str,
    eval_step,
    make_eval_batch,
    num_test: int,
    num_train: int,
    make_train_batch,
    c_dagger_sigma: float,
) -> float:
    header = ["C", "lr", "momentum", "wd", "test_acc", "test_loss", "steps"]
    best_C, best_acc = None, -1.0

    for C in C_grid:
        params, steps = train_full_dpsgd(
            model=model,
            params_init=params_init,
            epochs=1,
            logical_bs=logical_bs,
            physical_bs=physical_bs,
            lr=lr,
            wd=wd,
            momentum=momentum,
            sampler_name=sampler_name,
            seed=seed,
            subsample_seed=subsample_seed,
            C=float(C),
            sigma=c_dagger_sigma,
            num_train=num_train,
            make_train_batch=make_train_batch,
        )
        acc, loss = eval_full(params, eval_step, make_eval_batch, num_test, physical_bs)
        append_row(
            out_csv,
            header,
            [float(C), float(lr), float(momentum), float(wd), acc, loss, int(steps)],
        )
        if acc > best_acc:
            best_acc = acc
            best_C = float(C)

    return float(best_C)


# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=sorted(DATASET_REGISTRY.keys()),
    )
    p.add_argument(
        "--model", type=str, default="resnet18", choices=sorted(MODEL_REGISTRY.keys())
    )
    p.add_argument(
        "--sampler", type=str, default="reshuffle", choices=["reshuffle", "poisson"]
    )

    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--mini_bs", type=int, default=32)

    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4)

    p.add_argument("--epochs_list", nargs="+", type=int, default=[1, 10, 25, 50])

    p.add_argument("--C_min", type=float, default=0.05)
    p.add_argument("--C_max", type=float, default=5.0)
    p.add_argument("--C_steps", type=int, default=15)

    p.add_argument("--sigma_mults", nargs="+", type=float, default=[1.0])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample_seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="results_ccs_sigma_th_utility")

    p.add_argument("--best_hparams_template", type=str, default=None)
    p.add_argument(
        "--tx_preset", type=str, default="tiny_128", choices=sorted(TX_PRESETS.keys())
    )
    args = p.parse_args()

    tx_preset = TX_PRESETS[args.tx_preset]

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.model == "tx_small" or args.model == "vit_tiny_cifar":
        model_name = args.model + "_" + args.tx_preset
    else:
        model_name = args.model
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = (
        f"{args.dataset}_{model_name}_{args.sampler}_bs{args.bs}_mini_bs{args.mini_bs}_seed{args.seed}"
        f"_hparam_modeper_epoch"
    )
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_json(os.path.join(run_dir, "config.json"), vars(args))

    seed_plan: Dict[str, Any] = {
        "master_seed": int(args.seed),
        "subsample_seed": (
            int(args.subsample_seed) if args.subsample_seed is not None else None
        ),
        "init_seed": _derive_seed(int(args.seed), "init", args.dataset, args.model),
        "runs": [],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }

    dataset_cfg = DATASET_REGISTRY[args.dataset]
    batch_kind = dataset_cfg["batch_kind"]

    # Load arrays
    meta = {}
    arrays: Dict[str, Any] = {}

    if batch_kind == "vision":
        train_images_np, train_labels_np, test_images_np, test_labels_np = dataset_cfg[
            "array_loader"
        ]()
        arrays["train_images"] = jax.device_put(train_images_np)
        arrays["train_labels"] = jax.device_put(train_labels_np.astype(np.int32))
        arrays["test_images"] = jax.device_put(test_images_np)
        arrays["test_labels"] = jax.device_put(test_labels_np.astype(np.int32))
        num_train = int(train_images_np.shape[0])
        num_test = int(test_images_np.shape[0])
        input_shape = dataset_cfg["input_shape"]

    else:
        Xtr, Atr, ytr, Xte, Ate, yte, meta = load_agnews_arrays(
            max_len=int(tx_preset["max_len"]), vocab_size=int(tx_preset["vocab_size"])
        )
        meta = {
            **tx_preset,
        }
        arrays["train_tokens"] = jax.device_put(Xtr)
        arrays["train_attn"] = jax.device_put(Atr)
        arrays["train_labels"] = jax.device_put(ytr.astype(np.int32))
        arrays["test_tokens"] = jax.device_put(Xte)
        arrays["test_attn"] = jax.device_put(Ate)
        arrays["test_labels"] = jax.device_put(yte.astype(np.int32))
        num_train = int(Xtr.shape[0])
        num_test = int(Xte.shape[0])
        input_shape = (1, int(meta["max_len"]))  # tokens

    logical_bs = int(args.bs)
    physical_bs = int(args.mini_bs or args.bs)
    if physical_bs > logical_bs:
        raise ValueError("mini_bs cannot exceed bs")
    if logical_bs % physical_bs != 0:
        raise ValueError("bs must be divisible by mini_bs")

    # Build model
    model_builder = MODEL_REGISTRY[args.model]
    model = model_builder(dataset_cfg["num_classes"], meta)

    # Init params
    init_key = jax.random.PRNGKey(int(seed_plan["init_seed"]))
    if batch_kind == "vision":
        init_batch = jnp.ones(input_shape, dtype=jnp.float32)
        params_init = model.init(init_key, init_batch)["params"]
    else:
        T = int(meta["max_len"])
        init_tokens = jnp.zeros((1, T), dtype=jnp.int32)
        init_attn = jnp.ones((1, T), dtype=jnp.int32)
        params_init = model.init(init_key, init_tokens, init_attn, train=True)["params"]

    # batch fns
    make_train_batch, make_eval_batch, get_num_test = make_batch_fns(batch_kind, arrays)
    eval_step = make_eval_step(model)

    _, M_steps = make_sampler(args.sampler, num_train, logical_bs, epochs=1)
    sigma_th = compute_sigma_th(
        M_steps, sampler="shuffling" if args.sampler == "reshuffle" else "poisson"
    )

    append_row(
        os.path.join(run_dir, "summary.csv"),
        ["item", "value"],
        ["M_steps", int(M_steps)],
    )
    append_row(
        os.path.join(run_dir, "summary.csv"),
        ["item", "value"],
        ["sigma_th", float(sigma_th)],
    )
    fallback = (args.lr, args.momentum, args.wd)
    E_ref = 1

    if args.best_hparams_template:
        lr_ref, mom_ref, wd_ref = load_best_hparams_for_epoch(
            args.best_hparams_template, E_ref, int(args.bs), fallback
        )
    else:
        lr_ref, mom_ref, wd_ref = fallback

    hparams_by_E: Dict[int, Tuple[float, float, float]] = {}
    for E in args.epochs_list:
        if args.best_hparams_template:
            lrE, momE, wdE = load_best_hparams_for_epoch(
                args.best_hparams_template, int(E), int(args.bs), fallback
            )
        else:
            lrE, momE, wdE = fallback
        hparams_by_E[int(E)] = (lrE, momE, wdE)

    c_dagger_sigma = sigma_th

    C_grid = np.linspace(args.C_min, args.C_max, args.C_steps).tolist()
    csel_csv = os.path.join(run_dir, "C_select_clip_only_1epoch.csv")
    C_dag = select_C_dagger_one_epoch(
        model=model,
        params_init=params_init,
        logical_bs=logical_bs,
        physical_bs=physical_bs,
        lr=lr_ref,
        wd=wd_ref,
        momentum=mom_ref,
        sampler_name=args.sampler,
        seed=args.seed,
        subsample_seed=args.subsample_seed,
        C_grid=C_grid,
        out_csv=csel_csv,
        eval_step=eval_step,
        make_eval_batch=make_eval_batch,
        num_test=get_num_test(),
        num_train=num_train,
        make_train_batch=make_train_batch,
        c_dagger_sigma=c_dagger_sigma,
    )
    append_row(
        os.path.join(run_dir, "summary.csv"),
        ["item", "value"],
        ["C_dagger", float(C_dag)],
    )

    # Main results CSV
    out_csv = os.path.join(run_dir, "utility_sigma_th.csv")
    header = [
        "sampler",
        "epochs",
        "bs",
        "mini_bs",
        "lr",
        "momentum",
        "wd",
        "C_dagger",
        "sigma_th",
        "sigma_used",
        "mode",
        "steps",
        "test_acc",
        "test_loss",
    ]

    for E in args.epochs_list:
        lrE, momE, wdE = hparams_by_E[int(E)]

        # clean
        clean_seed = _derive_seed(
            int(seed_plan["master_seed"]),
            "clean",
            args.dataset,
            args.model,
            args.sampler,
            int(E),
            int(args.bs),
            int(physical_bs),
        )
        seed_plan["runs"].append(
            {
                "mode": "clean",
                "epochs": int(E),
                "sigma_mult": None,
                "seed": int(clean_seed),
            }
        )
        params_clean, steps_clean = train_clean_sgd(
            model=model,
            params_init=params_init,
            epochs=int(E),
            logical_bs=logical_bs,
            physical_bs=physical_bs,
            lr=lrE,
            wd=wdE,
            momentum=momE,
            sampler_name=args.sampler,
            seed=int(clean_seed),
            subsample_seed=args.subsample_seed,
            num_train=num_train,
            make_train_batch=make_train_batch,
        )
        acc, loss = eval_full(
            params_clean, eval_step, make_eval_batch, get_num_test(), physical_bs
        )
        append_row(
            out_csv,
            header,
            [
                args.sampler,
                int(E),
                logical_bs,
                physical_bs,
                lrE,
                momE,
                wdE,
                float(C_dag),
                float(sigma_th),
                0.0,
                "clean",
                int(steps_clean),
                float(acc),
                float(loss),
            ],
        )

        # DP at mult*sigma_th
        mult = 1
        sig = float(mult) * float(sigma_th)
        dp_seed = _derive_seed(
            int(seed_plan["master_seed"]),
            "dp",
            args.dataset,
            args.model,
            args.sampler,
            int(E),
            float(mult),
            int(args.bs),
            int(physical_bs),
        )
        seed_plan["runs"].append(
            {
                "mode": "dp",
                "epochs": int(E),
                "sigma_mult": float(mult),
                "seed": int(dp_seed),
            }
        )
        params_dp, steps_dp = train_full_dpsgd(
            model=model,
            params_init=params_init,
            epochs=int(E),
            logical_bs=logical_bs,
            physical_bs=physical_bs,
            lr=lrE,
            wd=wdE,
            momentum=momE,
            sampler_name=args.sampler,
            seed=int(dp_seed),
            subsample_seed=args.subsample_seed,
            C=float(C_dag),
            sigma=float(sig),
            num_train=num_train,
            make_train_batch=make_train_batch,
        )
        acc, loss = eval_full(
            params_dp, eval_step, make_eval_batch, get_num_test(), physical_bs
        )
        append_row(
            out_csv,
            header,
            [
                args.sampler,
                int(E),
                logical_bs,
                physical_bs,
                lrE,
                momE,
                wdE,
                float(C_dag),
                float(sigma_th),
                float(sig),
                f"dp_sigma_{mult:.3g}x",
                int(steps_dp),
                float(acc),
                float(loss),
            ],
        )

    save_json(os.path.join(run_dir, "seeds.json"), seed_plan)
    print("\nDONE. Results in:", run_dir)


if __name__ == "__main__":
    main()
