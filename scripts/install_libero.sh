#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIBERO_DIR="$ROOT/third_party/LIBERO"

if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  if [[ -e "$LIBERO_DIR" ]]; then
    echo "Refusing to replace existing non-Git path: $LIBERO_DIR" >&2
    exit 1
  fi
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

if [[ -n "${LIBERO_REF:-}" ]]; then
  git -C "$LIBERO_DIR" fetch --tags origin
  git -C "$LIBERO_DIR" checkout --detach "$LIBERO_REF"
fi

for project in openvla-oft VLA-Adapter; do
  link="$ROOT/$project/LIBERO"
  if [[ -L "$link" ]]; then
    continue
  fi
  if [[ -e "$link" ]]; then
    echo "Refusing to replace existing path: $link" >&2
    exit 1
  fi
  ln -s ../third_party/LIBERO "$link"
done

echo "LIBERO installed at $LIBERO_DIR"
echo "Install it into both Python environments with: pip install -e $LIBERO_DIR"
