# common.py
import jax
import jax.numpy as jnp
import flax.linen as nn
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any

# ----------------------------
# Text (AG News) utilities
# ----------------------------
from typing import Optional, Tuple, List, Dict


def _basic_tokenize(s: str) -> List[str]:
    return s.lower().replace("\n", " ").split()


def _try_load_agnews_raw() -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Load AG News as raw texts + integer labels.

    Tries torchtext first, then HuggingFace datasets.
    """
    # Prefer torchtext if available
    try:
        from torchtext.datasets import AG_NEWS  # type: ignore

        train_iter = AG_NEWS(split="train")
        test_iter = AG_NEWS(split="test")

        train_texts, train_labels = [], []
        for y, x in train_iter:
            train_labels.append(int(y) - 1)
            train_texts.append(x)

        test_texts, test_labels = [], []
        for y, x in test_iter:
            test_labels.append(int(y) - 1)
            test_texts.append(x)

        return (
            train_texts,
            np.array(train_labels, np.int32),
            test_texts,
            np.array(test_labels, np.int32),
        )
    except Exception:
        pass

    # Fallback to HF datasets if available
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("ag_news")
        train_texts = list(ds["train"]["text"])
        train_labels = np.array(ds["train"]["label"], dtype=np.int32)
        test_texts = list(ds["test"]["text"])
        test_labels = np.array(ds["test"]["label"], dtype=np.int32)
        return train_texts, train_labels, test_texts, test_labels
    except Exception as e:
        raise RuntimeError(
            "Could not load AG News. Install torchtext or datasets.\n"
            "pip install torchtext  OR  pip install datasets\n"
            f"Underlying error: {e}"
        )


def _build_vocab(
    texts: List[str], vocab_size: int, min_freq: int = 2
) -> Dict[str, int]:
    from collections import Counter

    c = Counter()
    for t in texts:
        c.update(_basic_tokenize(t))

    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
    for tok, freq in c.most_common():
        if freq < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= vocab_size:
            break
    return vocab


def _encode_texts(
    texts: List[str],
    labels: np.ndarray,
    vocab: Dict[str, int],
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(texts)
    T = int(max_len)
    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]
    cls_id = vocab["[CLS]"]

    tokens = np.full((N, T), pad_id, dtype=np.int32)
    attn = np.zeros((N, T), dtype=np.int32)

    for i, t in enumerate(texts):
        ids = [cls_id] + [vocab.get(tok, unk_id) for tok in _basic_tokenize(t)]
        ids = ids[:T]
        tokens[i, : len(ids)] = np.asarray(ids, dtype=np.int32)
        attn[i, : len(ids)] = 1

    return tokens, attn, labels.astype(np.int32)


def load_agnews_arrays(
    *,
    max_len: int = 128,
    vocab_size: int = 30000,
    min_freq: int = 2,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, int],
]:
    """Return token/attention arrays for AG News plus metadata."""
    train_texts, train_labels, test_texts, test_labels = _try_load_agnews_raw()
    vocab = _build_vocab(train_texts, vocab_size=vocab_size, min_freq=min_freq)
    Xtr, Atr, ytr = _encode_texts(train_texts, train_labels, vocab, max_len=max_len)
    Xte, Ate, yte = _encode_texts(test_texts, test_labels, vocab, max_len=max_len)
    meta = {"vocab_size": int(len(vocab)), "max_len": int(max_len)}
    return Xtr, Atr, ytr, Xte, Ate, yte, meta


# --- Lower Bound Sigma Calculation ---
def compute_sigma_th(M: int, sampler: str = "poisson") -> float:
    """
    Computes theoretical sigma based on number of steps M.
    sigma_th = 1.0 / sqrt(2.0 * ln(max(M, 2)))
    """
    if sampler == "shuffling":
        return 1.0 / math.sqrt(2.0 * math.log(max(M, 2)))
    if sampler == "poisson":
        return 1.0 / math.sqrt(2.0 * math.log(max(M, 2)))


class AdaptiveGroupNorm(nn.Module):
    """GroupNorm that automatically picks a valid number of groups."""

    max_groups: int = 32
    epsilon: float = 1e-6

    def _select_num_groups(self, channels: int) -> int:
        num_groups = min(self.max_groups, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return max(1, num_groups)

    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]
        num_groups = self._select_num_groups(channels)
        return nn.GroupNorm(num_groups=num_groups, epsilon=self.epsilon)(x)


# --- ResNet Block (Linen) ---
class ResNetBlock(nn.Module):
    features: int
    stride: int = 1
    norm_cls: Any = AdaptiveGroupNorm  # Default to GroupNorm for DP

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # Conv1
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=(1, 1),
            use_bias=False,
        )(x)
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            y = norm(y, use_running_average=not train)
        else:
            y = norm(y)
        y = nn.relu(y)

        # Conv2
        y = nn.Conv(
            self.features, kernel_size=(3, 3), strides=1, padding=(1, 1), use_bias=False
        )(y)
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            y = norm(y, use_running_average=not train)
        else:
            y = norm(y)

        # Downsample if needed
        if residual.shape != y.shape:
            residual = nn.Conv(
                self.features, kernel_size=(1, 1), strides=self.stride, use_bias=False
            )(residual)
            norm = self.norm_cls()
            if isinstance(norm, nn.BatchNorm):
                residual = norm(residual, use_running_average=not train)
            else:
                residual = norm(residual)

        return nn.relu(y + residual)


# --- ResNet18 for CIFAR-10 (Linen) ---
class ResNet18(nn.Module):
    num_classes: int = 10
    norm_cls: Any = AdaptiveGroupNorm  # Default to GroupNorm for DP

    @nn.compact
    def __call__(self, x, train: bool = True):
        # CIFAR Stem: 3x3 Conv, stride 1, no MaxPool
        x = nn.Conv(64, kernel_size=(3, 3), strides=1, padding=(1, 1), use_bias=False)(
            x
        )
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            x = norm(x, use_running_average=not train)
        else:
            x = norm(x)
        x = nn.relu(x)

        # Layers
        x = self._make_layer(x, 64, 2, stride=1, train=train)
        x = self._make_layer(x, 128, 2, stride=2, train=train)
        x = self._make_layer(x, 256, 2, stride=2, train=train)
        x = self._make_layer(x, 512, 2, stride=2, train=train)

        # Classifier
        x = jnp.mean(x, axis=(-3, -2))  # Global Average Pooling
        x = nn.Dense(self.num_classes)(x)
        return x

    def _make_layer(self, x, features, blocks, stride, train):
        # First block handles stride
        x = ResNetBlock(features, stride=stride, norm_cls=self.norm_cls)(x, train=train)
        # Subsequent blocks are stride 1
        for _ in range(1, blocks):
            x = ResNetBlock(features, stride=1, norm_cls=self.norm_cls)(x, train=train)
        return x


# --- ResNet34 (deeper baseline for CIFAR-100) ---
class ResNet34(nn.Module):
    num_classes: int = 100
    norm_cls: Any = AdaptiveGroupNorm

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, kernel_size=(3, 3), strides=1, padding=(1, 1), use_bias=False)(
            x
        )
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            x = norm(x, use_running_average=not train)
        else:
            x = norm(x)
        x = nn.relu(x)

        x = self._make_layer(x, 64, 3, stride=1, train=train)
        x = self._make_layer(x, 128, 4, stride=2, train=train)
        x = self._make_layer(x, 256, 6, stride=2, train=train)
        x = self._make_layer(x, 512, 3, stride=2, train=train)

        x = jnp.mean(x, axis=(-3, -2))
        x = nn.Dense(self.num_classes)(x)
        return x

    def _make_layer(self, x, features, blocks, stride, train):
        x = ResNetBlock(features, stride=stride, norm_cls=self.norm_cls)(x, train=train)
        for _ in range(1, blocks):
            x = ResNetBlock(features, stride=1, norm_cls=self.norm_cls)(x, train=train)
        return x


# --- Wide ResNet Blocks ---
class WideResNetBlock(nn.Module):
    features: int
    stride: int = 1
    dropout_rate: float = 0.0
    norm_cls: Any = AdaptiveGroupNorm

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=(1, 1),
            use_bias=False,
        )(x)
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            y = norm(y, use_running_average=not train)
        else:
            y = norm(y)
        y = nn.relu(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=1,
            padding=(1, 1),
            use_bias=False,
        )(y)
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            y = norm(y, use_running_average=not train)
        else:
            y = norm(y)

        if residual.shape != y.shape:
            residual = nn.Conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
            )(residual)
            norm = self.norm_cls()
            if isinstance(norm, nn.BatchNorm):
                residual = norm(residual, use_running_average=not train)
            else:
                residual = norm(residual)

        return nn.relu(y + residual)


class WideResNet(nn.Module):
    depth: int
    width: int
    num_classes: int
    dropout_rate: float = 0.0
    norm_cls: Any = AdaptiveGroupNorm

    @nn.compact
    def __call__(self, x, train: bool = True):
        if (self.depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should be 6n + 4.")
        n = (self.depth - 4) // 6

        x = nn.Conv(16, kernel_size=(3, 3), strides=1, padding=(1, 1), use_bias=False)(
            x
        )
        norm = self.norm_cls()
        if isinstance(norm, nn.BatchNorm):
            x = norm(x, use_running_average=not train)
        else:
            x = norm(x)
        x = nn.relu(x)

        widths = [16 * self.width, 32 * self.width, 64 * self.width]
        strides = [1, 2, 2]
        for features, stride in zip(widths, strides):
            for i in range(n):
                block_stride = stride if i == 0 else 1
                x = WideResNetBlock(
                    features=features,
                    stride=block_stride,
                    dropout_rate=self.dropout_rate,
                    norm_cls=self.norm_cls,
                )(x, train=train)

        x = jnp.mean(x, axis=(-3, -2))
        x = nn.Dense(self.num_classes)(x)
        return x


class WideResNet28x10(WideResNet):
    depth: int = 28
    width: int = 10
    num_classes: int = 10


# --- Data Loading ---
def get_cifar10_datasets(batch_size: int):
    """
    Load CIFAR-10 using TFDS.
    Returns (train_ds, test_ds, train_size, steps_per_epoch).
    Iterators yield dictionaries with 'image' (float32, 0-1) and 'label' (int32).
    """
    train_ds, ds_info = tfds.load(
        "cifar10", split="train", with_info=True, as_supervised=False
    )
    test_ds = tfds.load("cifar10", split="test", as_supervised=False)

    train_size = ds_info.splits["train"].num_examples

    def normalize_img(sample):
        image = tf.cast(sample["image"], tf.float32) / 255.0
        # Standard CIFAR mean/std normalization
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2470, 0.2435, 0.2616], dtype=tf.float32)
        image = (image - mean) / std
        return {"image": image, "label": sample["label"]}

    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(train_size)

    # drop_remainder=True is important for fixed batch size in JAX
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = train_size // batch_size

    return train_ds, test_ds, train_size, steps_per_epoch


def load_cifar10_arrays():
    """Load the entire CIFAR-10 dataset into NumPy arrays with CIFAR normalization."""
    train_ds = tfds.load("cifar10", split="train", as_supervised=False, batch_size=-1)
    test_ds = tfds.load("cifar10", split="test", as_supervised=False, batch_size=-1)

    train_np = tfds.as_numpy(train_ds)
    test_np = tfds.as_numpy(test_ds)

    def normalize(images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32) / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
        return (images - mean) / std

    train_images = normalize(train_np["image"])
    test_images = normalize(test_np["image"])

    train_labels = train_np["label"].astype(np.int32)
    test_labels = test_np["label"].astype(np.int32)

    return train_images, train_labels, test_images, test_labels


def get_cifar100_datasets(batch_size: int):
    """Load CIFAR-100 with TFDS and per-channel normalization."""
    train_ds, ds_info = tfds.load(
        "cifar100", split="train", with_info=True, as_supervised=False
    )
    test_ds = tfds.load("cifar100", split="test", as_supervised=False)

    train_size = ds_info.splits["train"].num_examples
    mean = tf.constant([0.5071, 0.4867, 0.4408], dtype=tf.float32)
    std = tf.constant([0.2675, 0.2565, 0.2761], dtype=tf.float32)

    def normalize(sample):
        image = tf.cast(sample["image"], tf.float32) / 255.0
        image = (image - mean) / std
        return {"image": image, "label": sample["label"]}

    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        train_ds.cache()
        .shuffle(train_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = train_size // batch_size
    return train_ds, test_ds, train_size, steps_per_epoch


def load_cifar100_arrays():
    """Load CIFAR-100 arrays with normalization."""
    train_ds = tfds.load("cifar100", split="train", as_supervised=False, batch_size=-1)
    test_ds = tfds.load("cifar100", split="test", as_supervised=False, batch_size=-1)

    train_np = tfds.as_numpy(train_ds)
    test_np = tfds.as_numpy(test_ds)

    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32).reshape(1, 1, 1, 3)

    def normalize(images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32) / 255.0
        return (images - mean) / std

    train_images = normalize(train_np["image"])
    test_images = normalize(test_np["image"])
    train_labels = train_np["label"].astype(np.int32)
    test_labels = test_np["label"].astype(np.int32)
    return train_images, train_labels, test_images, test_labels


def get_svhn_datasets(batch_size: int, include_extra: bool = True):
    """
    Load SVHN (cropped) with TFDS. Optionally includes the extra split for additional data.
    """
    train_split = "train+extra" if include_extra else "train"
    train_ds, ds_info = tfds.load(
        "svhn_cropped", split=train_split, with_info=True, as_supervised=False
    )
    test_ds = tfds.load("svhn_cropped", split="test", as_supervised=False)

    splits_to_count = ["train", "extra"] if include_extra else ["train"]
    train_size = sum(ds_info.splits[name].num_examples for name in splits_to_count)

    mean = tf.constant([0.4377, 0.4438, 0.4728], dtype=tf.float32)
    std = tf.constant([0.1980, 0.2010, 0.1970], dtype=tf.float32)

    def normalize(sample):
        image = tf.cast(sample["image"], tf.float32) / 255.0
        image = (image - mean) / std
        return {"image": image, "label": sample["label"]}

    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        train_ds.cache()
        .shuffle(train_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = train_size // batch_size
    return train_ds, test_ds, train_size, steps_per_epoch


def load_svhn_arrays(include_extra: bool = True):
    """Load SVHN arrays with normalization."""
    train_split = "train+extra" if include_extra else "train"
    train_ds = tfds.load(
        "svhn_cropped", split=train_split, as_supervised=False, batch_size=-1
    )
    test_ds = tfds.load(
        "svhn_cropped", split="test", as_supervised=False, batch_size=-1
    )

    train_np = tfds.as_numpy(train_ds)
    test_np = tfds.as_numpy(test_ds)

    mean = np.array([0.4377, 0.4438, 0.4728], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.1980, 0.2010, 0.1970], dtype=np.float32).reshape(1, 1, 1, 3)

    def normalize(images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32) / 255.0
        return (images - mean) / std

    train_images = normalize(train_np["image"])
    test_images = normalize(test_np["image"])
    train_labels = train_np["label"].astype(np.int32)
    test_labels = test_np["label"].astype(np.int32)
    return train_images, train_labels, test_images, test_labels
