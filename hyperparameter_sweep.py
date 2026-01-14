import argparse
import csv
import os
import hashlib
import random

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from common import (
    ResNet18,
    ResNet34,
    WideResNet28x10,
    get_cifar10_datasets,
    get_cifar100_datasets,
    get_svhn_datasets,
    load_agnews_arrays,
)

from models_text import TransformerConfig, TransformerClassifier, TX_PRESETS
from vit_models import ViTConfig, ViTClassifier, VIT_PRESETS


DATASET_REGISTRY = {
    "cifar10": {
        "num_classes": 10,
        "loader": get_cifar10_datasets,
        "input_shape": (1, 32, 32, 3),
        "display_name": "CIFAR10",
        "kind": "vision",
    },
    "cifar100": {
        "num_classes": 100,
        "loader": get_cifar100_datasets,
        "input_shape": (1, 32, 32, 3),
        "display_name": "CIFAR100",
        "kind": "vision",
    },
    "svhn": {
        "num_classes": 10,
        "loader": lambda bs: get_svhn_datasets(bs, include_extra=True),
        "input_shape": (1, 32, 32, 3),
        "display_name": "SVHN",
        "kind": "vision",
    },
    "agnews": {
        "num_classes": 4,
        "loader": None,
        "input_shape": None,
        "display_name": "AGNEWS",
        "kind": "text",
    },
}


MODEL_REGISTRY = {
    "resnet18": {
        "builder": lambda num_classes: ResNet18(num_classes=num_classes),
        "display_name": "ResNet18",
        "kind": "vision",
    },
    "resnet34": {
        "builder": lambda num_classes: ResNet34(num_classes=num_classes),
        "display_name": "ResNet34",
        "kind": "vision",
    },
    "wideresnet28x10": {
        "builder": lambda num_classes: WideResNet28x10(
            num_classes=num_classes,
            dropout_rate=0.0,
        ),
        "display_name": "WideResNet28x10",
        "kind": "vision",
    },
    "tx_small": {
        "builder": lambda num_classes, meta: TransformerClassifier(
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
        "display_name": "TransformerSmall",
        "kind": "text",
    },
    "vit_tiny_cifar": {
        "builder": lambda num_classes, meta=None: ViTClassifier(
            ViTConfig(
                num_classes=int(num_classes),
                image_size=32,
                dropout=0.0,
                attn_dropout=0.0,
                **VIT_PRESETS["vit_tiny_cifar"],
            )
        ),
        "display_name": "ViT Tiny CIFAR",
        "kind": "vision",
    },
    "vit_small_cifar": {
        "builder": lambda num_classes, meta=None: ViTClassifier(
            ViTConfig(
                num_classes=int(num_classes),
                image_size=32,
                dropout=0.0,
                attn_dropout=0.0,
                **VIT_PRESETS["vit_small_cifar"],
            )
        ),
        "display_name": "ViT Small CIFAR",
        "kind": "vision",
    },
    "vit_base_cifar": {
        "builder": lambda num_classes, meta=None: ViTClassifier(
            ViTConfig(
                num_classes=int(num_classes),
                image_size=32,
                dropout=0.0,
                attn_dropout=0.0,
                **VIT_PRESETS["vit_base_cifar"],
            )
        ),
        "display_name": "ViT Base CIFAR",
        "kind": "vision",
    },
}


def _derive_seed(master_seed: int, *tags) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(master_seed)).encode("utf-8"))
    for t in tags:
        h.update(b"|")
        h.update(str(t).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % (2**31 - 1)


def _iter_agnews_batches(
    *,
    tokens: np.ndarray,
    attn: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    epoch_seed: int,
    shuffle: bool,
):
    n = int(tokens.shape[0])
    idx = np.arange(n, dtype=np.int64)
    if shuffle:
        g = np.random.default_rng(int(epoch_seed))
        g.shuffle(idx)
    for s in range(0, n - (n % batch_size), batch_size):
        b = idx[s : s + batch_size]
        yield {
            "tokens": jnp.asarray(tokens[b]),
            "attn": jnp.asarray(attn[b]),
            "label": jnp.asarray(labels[b]),
        }


def _iter_agnews_eval_batches(
    *, tokens: np.ndarray, attn: np.ndarray, labels: np.ndarray, batch_size: int
):
    n = int(tokens.shape[0])
    for s in range(0, n - (n % batch_size), batch_size):
        e = s + batch_size
        yield {
            "tokens": jnp.asarray(tokens[s:e]),
            "attn": jnp.asarray(attn[s:e]),
            "label": jnp.asarray(labels[s:e]),
        }


def main():
    print(jax.config.read("jax_enable_x64"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=sorted(DATASET_REGISTRY.keys()),
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=sorted(MODEL_REGISTRY.keys())
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_list", nargs="+", type=float, default=[0.1, 0.05, 0.01])
    parser.add_argument("--momentum_list", nargs="+", type=float, default=[0.9])
    parser.add_argument("--wd_list", nargs="+", type=float, default=[5e-4])
    parser.add_argument("--out_dir", type=str, default="results_jax_sweep")
    parser.add_argument("--seed", type=int, default=0)

    # Text-specific (for agnews / tx_small)
    parser.add_argument("--text_max_len", type=int, default=128)
    parser.add_argument("--text_vocab_size", type=int, default=30000)
    parser.add_argument(
        "--tx_preset", type=str, default="tiny_128", choices=sorted(TX_PRESETS.keys())
    )
    args = parser.parse_args()

    tx_preset = TX_PRESETS[args.tx_preset]
    dataset_cfg = DATASET_REGISTRY[args.dataset]
    model_cfg = MODEL_REGISTRY[args.model]

    if dataset_cfg["kind"] != model_cfg["kind"]:
        raise ValueError(
            f"Incompatible dataset/model: dataset kind={dataset_cfg['kind']} vs model kind={model_cfg['kind']}"
        )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    model_tx = "_" + args.tx_preset if args.model == "tx_small" else ""
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(
        args.out_dir,
        f"sweep_{args.dataset}_{args.model}{model_tx}_bs{args.bs}_epochs{args.epochs}.csv",
    )
    best_csv = os.path.join(
        args.out_dir,
        f"best_{args.dataset}_{args.model}{model_tx}_epochs{args.epochs}.csv",
    )

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "model",
                "epochs",
                "bs",
                "lr",
                "momentum",
                "weight_decay",
                "test_acc",
                "test_loss",
            ]
        )

    meta = {}
    if dataset_cfg["kind"] == "vision":
        train_ds, test_ds, _, _ = dataset_cfg["loader"](args.bs)
        model = model_cfg["builder"](dataset_cfg["num_classes"])
        init_seed = _derive_seed(int(args.seed), "init", args.dataset, args.model)
        key = jax.random.PRNGKey(int(init_seed))
        init_variables = model.init(key, jnp.ones(dataset_cfg["input_shape"]))
        params = init_variables["params"]
        batch_stats = init_variables.get("batch_stats", {})
        use_batch_stats = bool(batch_stats)
    else:
        Xtr, Atr, ytr, Xte, Ate, yte, meta = load_agnews_arrays(
            max_len=int(tx_preset["max_len"]), vocab_size=int(tx_preset["vocab_size"])
        )
        meta = {
            **meta,
            "d_model": int(tx_preset["d_model"]),
            "n_heads": int(tx_preset["n_heads"]),
            "n_layers": int(tx_preset["n_layers"]),
            "d_ff": int(tx_preset["d_ff"]),
        }
        model = model_cfg["builder"](dataset_cfg["num_classes"], meta)
        init_seed = _derive_seed(
            int(args.seed), "init", args.dataset, args.model, int(tx_preset["max_len"])
        )
        key = jax.random.PRNGKey(int(init_seed))
        init_tokens = jnp.zeros((1, int(meta["max_len"])), dtype=jnp.int32)
        init_attn = jnp.ones((1, int(meta["max_len"])), dtype=jnp.int32)
        init_variables = model.init(key, init_tokens, init_attn, train=True)
        params = init_variables["params"]
        batch_stats = {}
        use_batch_stats = False

    best_result = None

    for lr in args.lr_list:
        for mom in args.momentum_list:
            for wd in args.wd_list:
                print(f"Running sweep: lr={lr}, mom={mom}, wd={wd}")
                is_existing = False
                if os.path.exists(out_csv):
                    with open(out_csv, "r") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if (
                                row[0] == args.dataset
                                and row[1] == args.model
                                and row[2] == args.epochs
                                and row[3] == args.bs
                                and row[4] == lr
                                and row[5] == mom
                                and row[6] == wd
                            ):
                                print(f"Already in CSV: lr={lr}, mom={mom}, wd={wd}")
                                is_existing = True
                                break
                if is_existing:
                    continue
                tx = optax.chain(
                    optax.add_decayed_weights(wd), optax.sgd(lr, momentum=mom)
                )
                opt_state = tx.init(params)

                # --- Train ---
                @jax.jit
                def train_step(params, batch_stats, opt_state, batch):
                    def loss_fn(p):
                        variables = {"params": p}
                        if use_batch_stats:
                            variables["batch_stats"] = batch_stats
                            logits, updates = model.apply(
                                variables,
                                batch["image"],
                                mutable=["batch_stats"],
                                train=True,
                            )
                            new_batch_stats = updates["batch_stats"]
                        else:
                            if dataset_cfg["kind"] == "vision":
                                logits = model.apply(
                                    variables, batch["image"], train=True
                                )
                            else:
                                logits = model.apply(
                                    variables,
                                    batch["tokens"],
                                    batch["attn"],
                                    train=True,
                                )
                            new_batch_stats = batch_stats
                        loss = optax.softmax_cross_entropy_with_integer_labels(
                            logits=logits, labels=batch["label"]
                        ).mean()
                        return loss, (logits, new_batch_stats)

                    (loss, (logits, new_batch_stats)), grads = jax.value_and_grad(
                        loss_fn, has_aux=True
                    )(params)

                    updates_opt, new_opt_state = tx.update(grads, opt_state, params)
                    new_params = optax.apply_updates(params, updates_opt)

                    acc = jnp.mean(jnp.argmax(logits, -1) == batch["label"])
                    return new_params, new_batch_stats, new_opt_state, loss, acc

                # --- Eval ---
                @jax.jit
                def eval_step(params, batch_stats, batch):
                    variables = {"params": params}
                    if use_batch_stats:
                        variables["batch_stats"] = batch_stats
                    if dataset_cfg["kind"] == "vision":
                        logits = model.apply(variables, batch["image"], train=False)
                    else:
                        logits = model.apply(
                            variables,
                            batch["tokens"],
                            batch["attn"],
                            train=False,
                        )
                    loss = optax.softmax_cross_entropy_with_integer_labels(
                        logits=logits, labels=batch["label"]
                    ).mean()
                    acc = jnp.mean(jnp.argmax(logits, -1) == batch["label"])
                    return {"loss": loss, "accuracy": acc}

                # --- Training Loop ---
                params_run = params
                batch_stats_run = batch_stats
                opt_state = tx.init(params_run)
                final_test_acc = None
                final_test_loss = None
                for epoch in range(args.epochs):
                    # Train
                    if dataset_cfg["kind"] == "vision":
                        for batch in train_ds.as_numpy_iterator():
                            params_run, batch_stats_run, opt_state, _, _ = train_step(
                                params_run, batch_stats_run, opt_state, batch
                            )
                    else:
                        epoch_seed = _derive_seed(
                            int(args.seed),
                            "train_epoch",
                            args.dataset,
                            args.model,
                            float(lr),
                            float(mom),
                            float(wd),
                            int(epoch),
                        )
                        for batch in _iter_agnews_batches(
                            tokens=Xtr,
                            attn=Atr,
                            labels=ytr,
                            batch_size=int(args.bs),
                            epoch_seed=int(epoch_seed),
                            shuffle=True,
                        ):
                            params_run, batch_stats_run, opt_state, _, _ = train_step(
                                params_run, batch_stats_run, opt_state, batch
                            )

                    # Eval
                    total_acc = 0.0
                    total_loss = 0.0
                    count = 0
                    if dataset_cfg["kind"] == "vision":
                        for batch in test_ds.as_numpy_iterator():
                            metrics = eval_step(params_run, batch_stats_run, batch)
                            total_acc += metrics["accuracy"]
                            total_loss += metrics["loss"]
                            count += 1
                    else:
                        for batch in _iter_agnews_eval_batches(
                            tokens=Xte,
                            attn=Ate,
                            labels=yte,
                            batch_size=int(args.bs),
                        ):
                            metrics = eval_step(params_run, batch_stats_run, batch)
                            total_acc += metrics["accuracy"]
                            total_loss += metrics["loss"]
                            count += 1

                    test_acc = float(total_acc / count)
                    test_loss = float(total_loss / count)
                    final_test_acc = test_acc
                    final_test_loss = test_loss

                    print(f"Result: acc={test_acc:.4f}, loss={test_loss:.4f}")

                    if best_result is None or test_acc > best_result["test_acc"]:
                        best_result = {
                            "dataset": dataset_cfg["display_name"],
                            "model": model_cfg["display_name"],
                            "epochs": args.epochs,
                            "bs": args.bs,
                            "lr": lr,
                            "momentum": mom,
                            "weight_decay": wd,
                            "test_acc": test_acc,
                            "test_loss": test_loss,
                        }
                if final_test_acc is not None and final_test_loss is not None:
                    with open(out_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                dataset_cfg["display_name"],
                                model_cfg["display_name"],
                                args.epochs,
                                args.bs,
                                lr,
                                mom,
                                wd,
                                final_test_acc,
                                final_test_loss,
                            ]
                        )

    if best_result is not None:
        header = [
            "dataset",
            "model",
            "epochs",
            "bs",
            "lr",
            "momentum",
            "weight_decay",
            "test_acc",
            "test_loss",
        ]
        write_header = not os.path.exists(best_csv)
        with open(best_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(
                [
                    best_result["dataset"],
                    best_result["model"],
                    best_result["epochs"],
                    best_result["bs"],
                    best_result["lr"],
                    best_result["momentum"],
                    best_result["weight_decay"],
                    best_result["test_acc"],
                    best_result["test_loss"],
                ]
            )


if __name__ == "__main__":
    main()
