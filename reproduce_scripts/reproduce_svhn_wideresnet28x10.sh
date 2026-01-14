#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_reproduce_common.sh"

# Setting: results/svhn_wideresnet28x10_bs{BS}
SETTING_NAME="svhn_wideresnet28x10"

BS="${1:-256}"
shift 1 || true

run_reproduce_setting "${SETTING_NAME}" "${BS}" "$@"


