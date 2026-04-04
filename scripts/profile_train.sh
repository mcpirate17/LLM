#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-profiles/latest}"
ARGS=("$@")

for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--output-dir" && $((i+1)) -lt ${#ARGS[@]} ]]; then
    OUTPUT_DIR="${ARGS[$((i+1))]}"
    break
  fi
done

python -m research.profiling.train_loop --output-dir "$OUTPUT_DIR" "${ARGS[@]}"

echo
echo "Optional Nsight commands:"
echo "  nsys profile -o ${OUTPUT_DIR}/raw/nsys_report --force-overwrite=true python -m research.profiling.train_loop --disable-torch-profiler --output-dir ${OUTPUT_DIR}/nsys --data-mode corpus --steps 12"
echo "  ncu --set full --target-processes all --export ${OUTPUT_DIR}/raw/ncu_report python -m research.profiling.train_loop --disable-torch-profiler --output-dir ${OUTPUT_DIR}/ncu --data-mode corpus --steps 12"
