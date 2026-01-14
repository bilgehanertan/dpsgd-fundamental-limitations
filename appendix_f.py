#!/usr/bin/env python3
"""
FIGURE 1 (final explicit bound):
  RHS(M; s in s_list) vs M

FIGURE 2 (real prediction):
  kappa_muGDP(M; E in Es, sigma(M)=s_pred/sqrt(ln M)) vs M

Outputs:
  figs/real_prediction_vs_M.pdf + .png
  figs/final_bound_vs_M.pdf + .png
"""

from __future__ import annotations

import os
import math
import numpy as np
import matplotlib.pyplot as plt


try:
    from scipy.stats import norm  # type: ignore

    def Phi(x: np.ndarray) -> np.ndarray:
        return norm.cdf(x)

except Exception:

    def Phi(x: np.ndarray) -> np.ndarray:
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def make_M_grid(M_min: int, M_max: int, num: int) -> np.ndarray:
    M = np.unique(
        np.round(np.logspace(np.log10(max(M_min, 2)), np.log10(M_max), num)).astype(int)
    )
    return M[M >= 2].astype(float)


def kappa_LB(M: np.ndarray) -> np.ndarray:
    """kappa_LB(M) = 1/sqrt(8) * (1 - 1/sqrt(4 pi ln M))"""
    return (1.0 / np.sqrt(8.0)) * (1.0 - 1.0 / np.sqrt(4.0 * np.pi * np.log(M)))


def sigma_schedule(M: np.ndarray, s: float) -> np.ndarray:
    """sigma(M) = s / sqrt(ln M)."""
    return s / np.sqrt(np.log(M))


def mu_gdp_asymptotic(M: np.ndarray, E: float, sigma: np.ndarray) -> np.ndarray:
    """
    mu = sqrt(2)*sqrt(E/M)*sqrt( exp(sigma^{-2}) Phi(1.5 sigma^{-1})
                               + 3 Phi(-0.5 sigma^{-1}) - 2 )
    """
    inv = 1.0 / sigma
    expo = np.exp(np.clip(inv**2, a_min=None, a_max=700.0))
    inside = expo * Phi(1.5 * inv) + 3.0 * Phi(-0.5 * inv) - 2.0
    inside = np.maximum(inside, 0.0)
    return np.sqrt(2.0) * np.sqrt(E / M) * np.sqrt(inside)


def kappa_from_mu(mu: np.ndarray) -> np.ndarray:
    """sep(G_mu) = (2 Phi(mu/2) - 1)/sqrt(2)."""
    return (2.0 * Phi(0.5 * mu) - 1.0) / np.sqrt(2.0)


def kappa_real_prediction(M: np.ndarray, s: float, E: float) -> np.ndarray:
    sig = sigma_schedule(M, s)
    mu = mu_gdp_asymptotic(M, E, sig)
    return np.clip(kappa_from_mu(mu), 0.0, 1.0 / np.sqrt(2.0))


def kappa_final_bound_rhs_unclipped(M: np.ndarray, s: float) -> np.ndarray:
    """
    Implements Eq. 52 in the paper:

    1/sqrt(2)
    - (2/sqrt(pi)) * exp(-(1/16) * M^{1/s^2 - 1})
      / ((1/sqrt(2)) * M^{1/(2s^2) - 1/2})
    """
    term1 = 1.0 / np.sqrt(2.0)

    numerator = (2.0 / np.sqrt(np.pi)) * np.exp(
        -(1.0 / 16.0) * (M ** (1.0 / (s**2) - 1.0))
    )

    denominator = (1.0 / np.sqrt(2.0)) * (M ** (1.0 / (2.0 * s**2) - 0.5))

    return term1 - numerator / denominator


def is_close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def sigma_label_from_s(s: float) -> str:
    """
    - if s = 1/sqrt(2): sigma = 1/sqrt(2 ln M)
    - else: sigma = s/sqrt(ln M)
    """
    s_th = 1.0 / np.sqrt(2.0)
    if is_close(s, float(s_th), tol=1e-9):
        return r"$\sigma=\frac{1}{\sqrt{2\ln M}}$"
    return rf"$\sigma=\frac{{{s:.3g}}}{{\sqrt{{\ln M}}}}$"


def set_rcparams(fontsize: int = 8) -> None:
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "legend.fontsize": fontsize - 2,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def savefig_all(fig: plt.Figure, outbase: str) -> None:
    fig.tight_layout(pad=0.2)
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# FIGURE 2: real prediction (multi-E)
# ----------------------------
def plot_real_prediction_multiE(
    M: np.ndarray, s_pred: float, Es: list[float], outdir: str
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.set_xscale("log")

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "D", "^", "v", "x", "+"]

    for i, E in enumerate(Es):
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]
        ax.plot(
            M,
            kappa_real_prediction(M, s=s_pred, E=E),
            linewidth=1.0,
            linestyle=ls,
            label=rf"$E={E:g}$",
        )

    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"separation $\kappa$")
    ax.set_title(rf"$\mu$-GDP prediction ({sigma_label_from_s(s_pred)})")
    ax.set_ylim(0.0, (1.0 / np.sqrt(2.0)) * 1.02)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)

    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.85,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.6,
        handletextpad=0.4,
    )

    savefig_all(fig, os.path.join(outdir, "real_prediction_vs_M"))


# ----------------------------
# FIGURE 1: final explicit bound (multi-s)
# ----------------------------
def plot_final_bound_multiS(M: np.ndarray, s_list: list[float], outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.set_xscale("log")

    for s in s_list:
        rhs = kappa_final_bound_rhs_unclipped(M, s=s)
        rhs_plot = np.where(rhs >= 0.0, rhs, np.nan)
        ax.plot(M, rhs_plot, linewidth=1.0, label=sigma_label_from_s(s))

    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"separation $\kappa$")
    ax.set_title(r"Tail bound (E=1)")
    ax.set_ylim(0.0, (1.0 / np.sqrt(2.0)) * 1.02)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)

    ax.legend(
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.85,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.6,
        handletextpad=0.4,
    )

    savefig_all(fig, os.path.join(outdir, "final_bound_vs_M"))


def main() -> None:
    outdir = "figs"
    os.makedirs(outdir, exist_ok=True)
    set_rcparams(fontsize=8)

    M_R = make_M_grid(M_min=3, M_max=200_000, num=300)
    M_F = make_M_grid(M_min=1000, M_max=200_000, num=300)

    s_pred = 1.0 / np.sqrt(2.0)
    Es = [1.0, 5.0, 10.0, 25.0, 50.0]

    s_list = [0.40, 0.50, 0.60, 1.0 / np.sqrt(2.0), 0.80, 0.90]

    plot_final_bound_multiS(M_F, s_list=s_list, outdir=outdir)
    plot_real_prediction_multiE(M_R, s_pred=s_pred, Es=Es, outdir=outdir)

    print(f"[done] wrote figures to: {outdir}")
    print(f"  {outdir}/real_prediction_vs_M.pdf/.png")
    print(f"  {outdir}/final_bound_vs_M.pdf/.png")


if __name__ == "__main__":
    main()
