#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_reproduce_common.sh"

# Setting: results/cifar10_vit_tiny_cifar_bs{BS}
SETTING_NAME="cifar10_vit_tiny_cifar"

BS="${1:-256}"
shift 1 || true

run_reproduce_setting "${SETTING_NAME}" "${BS}" "$@"


