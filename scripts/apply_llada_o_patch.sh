#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$repo_root/third_party/llada_o"
patch="$repo_root/integrations/llada_o/patches/0001-satome-t2i-hooks.patch"

git -C "$repo_root" submodule update --init third_party/llada_o

if git -C "$upstream" apply --reverse --check "$patch" >/dev/null 2>&1; then
  echo "LLaDA-o SaToMe patch is already applied."
elif git -C "$upstream" apply --check "$patch"; then
  git -C "$upstream" apply "$patch"
  echo "Applied SaToMe T2I hooks to third_party/llada_o."
else
  echo "Cannot apply SaToMe patch: the LLaDA-o worktree differs from the pinned revision." >&2
  exit 1
fi
