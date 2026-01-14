#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
ROOT_DEFAULT="${REPO_ROOT_DEFAULT}"

usage_reproduce_common() {
  cat <<'EOF'
Usage:
  ./reproduce_scripts/reproduce_<SETTING>.sh [BS] [extra reproduce.py args...]

Notes:
  - BS defaults to 256.
  - mini_bs is fixed to 32 (the scripts match run directories containing `mini_bs32`).
  - Any extra args are passed through to reproduce.py (e.g., --only-epochs 1).
EOF
}

run_reproduce_setting() {
  local setting_name="$1"  # e.g. cifar10_resnet18
  local bs="${2:-256}"
  shift 2 || true

  local root="${ROOT:-$ROOT_DEFAULT}"

  local results_dir="${root}/results/${setting_name}_bs${bs}"

  if [[ ! -d "${results_dir}" ]]; then
    echo "[reproduce] results dir not found: ${results_dir}" >&2
    return 2
  fi

  local mini_glob="mini_bs32"

  local -a run_dirs=()
  local d
  for d in "${results_dir}"/*_poisson_bs${bs}_${mini_glob}_seed*_hparam_mode*; do
    [[ -d "${d}" ]] && run_dirs+=("${d}")
  done
  for d in "${results_dir}"/*_reshuffle_bs${bs}_${mini_glob}_seed*_hparam_mode*; do
    [[ -d "${d}" ]] && run_dirs+=("${d}")
  done

  if [[ "${#run_dirs[@]}" -eq 0 ]]; then
    echo "[reproduce] no run dirs found under ${results_dir} for bs=${bs} ${mini_glob}" >&2
    echo "  expected patterns:" >&2
    echo "    *_poisson_bs${bs}_${mini_glob}_seed*_hparam_mode*" >&2
    echo "    *_reshuffle_bs${bs}_${mini_glob}_seed*_hparam_mode*" >&2
    return 3
  fi

  echo "[reproduce] setting=${setting_name}"
  echo "[reproduce] results_dir=${results_dir}"
  echo "[reproduce] run_dirs (${#run_dirs[@]}):"
  printf '  - %s\n' "${run_dirs[@]}"

  python3 "${root}/reproduce.py" "${run_dirs[@]}" "$@"
}


