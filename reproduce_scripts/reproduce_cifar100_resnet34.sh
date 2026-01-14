#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_reproduce_common.sh"

# Setting: results/cifar100_resnet34_bs{BS}
SETTING_NAME="cifar100_resnet34"

BS="${1:-256}"
shift 1 || true

run_reproduce_setting "${SETTING_NAME}" "${BS}" "$@"


