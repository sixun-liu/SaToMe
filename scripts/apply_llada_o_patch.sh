#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$repo_root/third_party/llada_o"
patch="$repo_root/integrations/llada_o/patches/0001-satome-t2i-hooks.patch"
check_only="${1:-}"

if [[ "$check_only" != "" && "$check_only" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

git -C "$repo_root" submodule update --init third_party/llada_o

if git -C "$upstream" apply --reverse --check "$patch" >/dev/null 2>&1; then
  echo "LLaDA-o SaToMe patch is already applied."
  if [[ "$check_only" == "--check" ]]; then
    python - "$upstream/modeling/lladao/lladao.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("LLaDA-o SaToMe patch syntax: PASS")
PY
  fi
elif git -C "$upstream" apply --check "$patch"; then
  if [[ "$check_only" == "--check" ]]; then
    if ! git -C "$upstream" diff --quiet || ! git -C "$upstream" diff --cached --quiet; then
      echo "Cannot run patch check with unrelated changes in third_party/llada_o." >&2
      exit 1
    fi
    cleanup() {
      git -C "$upstream" apply --reverse "$patch" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    git -C "$upstream" apply "$patch"
    python - "$upstream/modeling/lladao/lladao.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("LLaDA-o SaToMe patch apply/reverse: PASS")
PY
  else
    git -C "$upstream" apply "$patch"
    echo "Applied SaToMe T2I hooks to third_party/llada_o."
  fi
else
  echo "Cannot apply SaToMe patch: the LLaDA-o worktree differs from the pinned revision." >&2
  exit 1
fi
