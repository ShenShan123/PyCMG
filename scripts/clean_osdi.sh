#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

rm -rf "${BUILD_DIR}/osdi"
echo "Removed ${BUILD_DIR}/osdi"
