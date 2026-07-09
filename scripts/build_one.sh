#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <my_first_cute|cute_wmma>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cutlass_dir="${CUTLASS_DIR:-${repo_root}/third_party/cutlass}"
target="$1"

if [[ ! -d "${cutlass_dir}/include/cute" ]]; then
  echo "CUTLASS include directory not found: ${cutlass_dir}/include/cute" >&2
  echo "Set CUTLASS_DIR=/path/to/cutlass if you want to use another checkout." >&2
  exit 1
fi

case "${target}" in
  my_first_cute)
    src="${repo_root}/csrc/kernel/my_first_cute.cu"
    arch_flags=()
    ;;
  cute_wmma)
    src="${repo_root}/csrc/kernel/cute_wmma.cu"
    arch_flags=(-arch=sm_80)
    ;;
  *)
    echo "unknown target: ${target}" >&2
    echo "known targets: my_first_cute cute_wmma" >&2
    exit 2
    ;;
esac

nvcc -std=c++17 --expt-relaxed-constexpr \
  -I"${repo_root}/csrc" \
  -I"${cutlass_dir}/include" \
  "${arch_flags[@]}" \
  "${src}" \
  -o "${repo_root}/${target}"
