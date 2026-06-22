#!/usr/bin/env bash
# RQ3 refresh: budget x horizon sensitivity for the neural Double-DQN on the held-out test split.
set -euo pipefail
cd "$(dirname "$0")/.."
export GEMINI_API_KEY="$(tr -d '\r\n' < "$HOME/Desktop/keys/gemini.md")"
P=gemini; M=gemini-2.5-flash-lite; EXP=dqn_eval; PY="python3 -m testgenrl.cli"
SUBJ="$EXP/subjects"; CACHE="$EXP/.llm_cache"
mkdir -p "$EXP/configs"
for b in low:300 medium:1000 high:2000; do
  name=${b%%:*}; tok=${b##*:}
  for h in 8 16 32; do
    cfg="$EXP/configs/b_${name}_h${h}.yaml"
    printf 'tokens: %s\nruntime_seconds: 30\nfuzz_iterations: 100\nhorizon: %s\n' "$tok" "$h" > "$cfg"
    for s in 0 1 2; do
      $PY eval --model "$EXP/models/dqn_s$s.json" --subjects "$SUBJ" --split test --budget "$cfg" \
        --llm-provider "$P" --llm-model "$M" --cache-mode replay --cache-dir "$CACHE" --seed "$s" \
        --out "$EXP/rq3/${name}_h${h}" >/dev/null
    done
  done
done
echo "RQ3 sweep done"
