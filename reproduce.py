#!/usr/bin/env python3
"""
reproduce.py

Reproduce an existing run_dp run directory using the saved hyperparameters, clipping
constant (C_dagger), sigma, and derived seeds (including init_seed),
and write a reproduction CSV that can be used to regenerate the tables in the paper.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp

import run_dp


@dataclass(frozen=True)
class SeedKey:
    mode: str  # "clean" | "dp"
    epochs: int
    sigma_mult: Optional[float]  # None for clean/clip


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_summary_csv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].strip() == "item":
                continue
            if len(row) >= 2:
                out[row[0].strip()] = row[1].strip()
    return out


def _iter_run_dirs(paths: List[str]) -> List[str]:
    run_dirs: List[str] = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            raise ValueError(f"Expected directory path, got file: {p}")
        if os.path.exists(os.path.join(p, "config.json")):
            run_dirs.append(p)
            continue
        if not os.path.isdir(p):
            raise ValueError(f"Path not found: {p}")
        for name in sorted(os.listdir(p)):
            sub = os.path.join(p, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "config.json")):
                run_dirs.append(sub)
    seen = set()
    uniq: List[str] = []
    for d in run_dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _load_seed_map(
    seeds_json: Dict[str, Any],
) -> Tuple[int, Optional[int], Dict[SeedKey, int]]:
    init_seed = int(seeds_json["init_seed"])
    subsample_seed = seeds_json.get("subsample_seed", None)
    subsample_seed = None if subsample_seed is None else int(subsample_seed)

    seed_map: Dict[SeedKey, int] = {}
    for r in seeds_json.get("runs", []):
        mode = str(r["mode"])
        epochs = int(r["epochs"])
        sigma_mult = r.get("sigma_mult", None)
        sigma_mult = None if sigma_mult is None else float(sigma_mult)
        seed = int(r["seed"])
        seed_map[SeedKey(mode=mode, epochs=epochs, sigma_mult=sigma_mult)] = seed
    return init_seed, subsample_seed, seed_map


def _build_arrays_and_meta(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, int, int, Tuple[int, ...]]:
    dataset = str(cfg["dataset"])
    dataset_cfg = run_dp.DATASET_REGISTRY[dataset]
    batch_kind = dataset_cfg["batch_kind"]

    meta: Dict[str, Any] = {}
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
        input_shape = tuple(int(x) for x in dataset_cfg["input_shape"])
        return arrays, meta, batch_kind, num_train, num_test, input_shape

    # text (agnews)
    tx_preset = run_dp.TX_PRESETS[str(cfg["tx_preset"])]
    Xtr, Atr, ytr, Xte, Ate, yte, _ = run_dp.load_agnews_arrays(
        max_len=int(tx_preset["max_len"]), vocab_size=int(tx_preset["vocab_size"])
    )
    meta = {**tx_preset}
    arrays["train_tokens"] = jax.device_put(Xtr)
    arrays["train_attn"] = jax.device_put(Atr)
    arrays["train_labels"] = jax.device_put(ytr.astype(np.int32))
    arrays["test_tokens"] = jax.device_put(Xte)
    arrays["test_attn"] = jax.device_put(Ate)
    arrays["test_labels"] = jax.device_put(yte.astype(np.int32))
    num_train = int(Xtr.shape[0])
    num_test = int(Xte.shape[0])
    input_shape = (1, int(meta["max_len"]))
    return arrays, meta, batch_kind, num_train, num_test, input_shape


def _init_params(
    *,
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
    batch_kind: str,
    input_shape: Tuple[int, ...],
    init_seed: int,
):
    dataset = str(cfg["dataset"])
    dataset_cfg = run_dp.DATASET_REGISTRY[dataset]
    model_builder = run_dp.MODEL_REGISTRY[str(cfg["model"])]
    model = model_builder(dataset_cfg["num_classes"], meta)

    init_key = jax.random.PRNGKey(int(init_seed))
    if batch_kind == "vision":
        init_batch = jnp.ones(input_shape, dtype=jnp.float32)
        params_init = model.init(init_key, init_batch)["params"]
    else:
        t = int(meta["max_len"])
        init_tokens = jnp.zeros((1, t), dtype=jnp.int32)
        init_attn = jnp.ones((1, t), dtype=jnp.int32)
        params_init = model.init(init_key, init_tokens, init_attn, train=True)["params"]
    return model, params_init


def _float_eq(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= float(tol)


def reproduce_run_dir(
    *,
    run_dir: str,
    out_subdir: str,
    only_epochs: Optional[List[int]],
    only_modes: Optional[List[str]],
) -> None:
    run_dir = os.path.abspath(run_dir)
    cfg_path = os.path.join(run_dir, "config.json")
    seeds_path = os.path.join(run_dir, "seeds.json")
    summary_path = os.path.join(run_dir, "summary.csv")
    util_path = os.path.join(run_dir, "utility_sigma_th.csv")

    for p in (cfg_path, seeds_path, summary_path, util_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    cfg = _read_json(cfg_path)
    seeds_json = _read_json(seeds_path)
    summary = _read_summary_csv(summary_path)

    init_seed, subsample_seed, seed_map = _load_seed_map(seeds_json)

    print(f"[reproduce] loading run: {run_dir}", flush=True)
    print(
        f"  dataset={cfg.get('dataset')} model={cfg.get('model')} sampler={cfg.get('sampler')} "
        f"bs={cfg.get('bs')} mini_bs={cfg.get('mini_bs')}",
        flush=True,
    )

    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    arrays, meta, batch_kind, num_train, num_test, input_shape = _build_arrays_and_meta(
        cfg
    )

    logical_bs = int(cfg["bs"])
    physical_bs = int(cfg.get("mini_bs") or cfg["bs"])
    if physical_bs > logical_bs or (logical_bs % physical_bs != 0):
        raise ValueError(
            f"Invalid bs/mini_bs in {cfg_path}: bs={logical_bs}, mini_bs={physical_bs}"
        )

    model, params_init = _init_params(
        cfg=cfg,
        meta=meta,
        batch_kind=batch_kind,
        input_shape=input_shape,
        init_seed=init_seed,
    )
    make_train_batch, make_eval_batch, get_num_test = run_dp.make_batch_fns(
        batch_kind, arrays
    )
    eval_step = run_dp.make_eval_step(model)

    c_dagger_summary = float(summary.get("C_dagger", "nan"))
    sigma_th_summary = float(summary.get("sigma_th", "nan"))

    out_dir = os.path.join(run_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "utility_sigma_th_reproduced.csv")

    header_repro = [
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

    header = header_repro
    total_rows_seen = 0
    total_rows_written = 0

    with (
        open(util_path, "r", newline="") as f_in,
        open(out_csv, "w", newline="", buffering=1) as f_out,
    ):
        r = csv.DictReader(f_in)
        w = csv.writer(f_out)
        w.writerow(header)
        f_out.flush()
        os.fsync(f_out.fileno())

        for row in r:
            total_rows_seen += 1

            epochs = int(row["epochs"])
            mode = str(row["mode"])
            if only_epochs is not None and epochs not in set(
                int(x) for x in only_epochs
            ):
                continue
            if only_modes is not None and mode not in set(only_modes):
                continue

            print(
                f"[reproduce] row epochs={epochs} mode={mode}",
                flush=True,
            )

            sampler = str(row["sampler"])
            if sampler != str(cfg["sampler"]):
                raise ValueError(
                    f"Sampler mismatch: csv={sampler}, cfg={cfg['sampler']}"
                )

            lr = float(row["lr"])
            momentum = float(row["momentum"])
            wd = float(row["wd"])
            c_dagger = float(row["C_dagger"])
            sigma_th = float(row["sigma_th"])
            sigma_used = float(row["sigma_used"])

            if np.isfinite(c_dagger_summary) and not _float_eq(
                c_dagger, c_dagger_summary, 1e-12
            ):
                raise ValueError(f"C_dagger mismatch summary vs utility in {run_dir}")
            if np.isfinite(sigma_th_summary) and not _float_eq(
                sigma_th, sigma_th_summary, 1e-12
            ):
                raise ValueError(f"sigma_th mismatch summary vs utility in {run_dir}")

            if mode == "clean":
                seed_key = SeedKey(mode="clean", epochs=epochs, sigma_mult=None)
                run_seed = seed_map[seed_key]
                params, steps = run_dp.train_clean_sgd(
                    model=model,
                    params_init=params_init,
                    epochs=epochs,
                    logical_bs=logical_bs,
                    physical_bs=physical_bs,
                    lr=lr,
                    wd=wd,
                    momentum=momentum,
                    sampler_name=str(cfg["sampler"]),
                    seed=int(run_seed),
                    subsample_seed=subsample_seed,
                    num_train=num_train,
                    make_train_batch=make_train_batch,
                )
            else:
                sigma_mult = 1
                if sigma_mult is None:
                    raise ValueError(f"Unrecognized mode string: {mode}")
                seed_key = SeedKey(
                    mode="dp", epochs=epochs, sigma_mult=float(sigma_mult)
                )
                run_seed = seed_map[seed_key]
                params, steps = run_dp.train_full_dpsgd(
                    model=model,
                    params_init=params_init,
                    epochs=epochs,
                    logical_bs=logical_bs,
                    physical_bs=physical_bs,
                    lr=lr,
                    wd=wd,
                    momentum=momentum,
                    sampler_name=str(cfg["sampler"]),
                    seed=int(run_seed),
                    subsample_seed=subsample_seed,
                    C=c_dagger,
                    sigma=sigma_used,
                    num_train=num_train,
                    make_train_batch=make_train_batch,
                )

            acc_repro, loss_repro = run_dp.eval_full(
                params, eval_step, make_eval_batch, get_num_test(), physical_bs
            )

            out_row = [
                sampler,
                epochs,
                logical_bs,
                physical_bs,
                lr,
                momentum,
                wd,
                c_dagger,
                sigma_th,
                sigma_used,
                mode,
                int(steps),
                float(acc_repro),
                float(loss_repro),
            ]

            w.writerow(out_row)
            total_rows_written += 1
            print(
                f"[reproduce] wrote row epochs={epochs} mode={mode} (steps={int(steps)})",
                flush=True,
            )
            f_out.flush()
            os.fsync(f_out.fileno())

    report = {
        "run_dir": run_dir,
        "out_csv": out_csv,
        "rows_seen": total_rows_seen,
        "rows_written": total_rows_written,
    }
    with open(os.path.join(out_dir, "reproduce_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[reproduce] finished: {run_dir}", flush=True)
    print(f"  wrote: {out_csv}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="One or more run_dp run directories (containing config.json) or parent folders of runs.",
    )
    p.add_argument("--out-subdir", type=str, default="reproduce")
    p.add_argument("--only-epochs", nargs="+", type=int, default=None)
    p.add_argument("--only-modes", nargs="+", type=str, default=None)
    args = p.parse_args()

    run_dirs = _iter_run_dirs(list(args.paths))
    if not run_dirs:
        raise ValueError(
            "No run directories found. Provide a run dir (with config.json) or a parent folder."
        )

    for d in run_dirs:
        reproduce_run_dir(
            run_dir=d,
            out_subdir=str(args.out_subdir),
            only_epochs=args.only_epochs,
            only_modes=args.only_modes,
        )


if __name__ == "__main__":
    main()
