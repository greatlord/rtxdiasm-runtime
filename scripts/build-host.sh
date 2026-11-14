#!/usr/bin/env bash
set -euo pipefail
# Build rtxdisam_host for the current Linux target.
# Usage: ./scripts/build-host.sh [build-dir] [install-prefix]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build/host}"
PREFIX="${2:-$ROOT/artifacts/linux-$(uname -m)/host}"
cmake -S "$ROOT/src/rtxdisam.host" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$BUILD_DIR" --config Release -j
cmake --install "$BUILD_DIR" --config Release 2>/dev/null || true
echo "Host output prefix: $PREFIX"
