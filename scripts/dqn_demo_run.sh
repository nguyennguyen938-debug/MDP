#!/usr/bin/env bash
# End-to-end real-provider demo: generate logs -> train Double-DQN -> replay-eval -> report.
# Uses Gemini (gemini-2.5-flash-lite). Prompts are subject-only, so replay eval never cache-misses.
set -euo pipefail

cd "$(dirname "$0")/.."   # artifact/
export GEMINI_API_KEY="$(tr -d '\r\n' < "$HOME/Desktop/keys/gemini.md")"

PROVIDER=gemini
MODEL=gemini-2.5-flash-lite
DEMO=dqn_demo
PY="python3 -m testgenrl.cli"

rm -rf "$DEMO"
mkdir -p "$DEMO/configs"
printf 'tokens: 1000\nruntime_seconds: 30\nfuzz_iterations: 100\nhorizon: 16\n' > "$DEMO/configs/budget_medium.yaml"
BUD="$DEMO/configs/budget_medium.yaml"
SUBJ="$DEMO/subjects"
CACHE="$DEMO/.llm_cache"

echo "== prepare 10 real subjects (full_pilot: humaneval/mbpp/bugsinpy) =="
$PY prepare --dataset full_pilot --limit 10 --out "$SUBJ"

echo "== live log generation (populates LLM cache + replay buffers) =="
for p in existing fixed greedy bandit; do
  $PY run --planner "$p" --subjects "$SUBJ" --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode live --cache-dir "$CACHE" --out "$DEMO/runs"
done
for s in 0 1 2; do
  $PY run --planner random --subjects "$SUBJ" --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode live --cache-dir "$CACHE" --seed "$s" --out "$DEMO/runs"
done

echo "== train Double-DQN from the generated replay buffers =="
$PY train --runs "$DEMO/runs" --out "$DEMO/models/dqn.json" --gamma 0.95 --steps 3000 --seed 0

echo "== replay eval: baselines + trained DQN (deterministic, cached provider) =="
for p in existing fixed greedy bandit; do
  $PY run --planner "$p" --subjects "$SUBJ" --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --out "$DEMO/results"
done
for s in 0 1 2; do
  $PY run --planner random --subjects "$SUBJ" --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --seed "$s" --out "$DEMO/results"
done
$PY eval --model "$DEMO/models/dqn.json" --subjects "$SUBJ" --budget "$BUD" \
  --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --out "$DEMO/results"

echo "== report (replay results only -> consistent comparison) =="
$PY report --runs "$DEMO/results" --out "$DEMO/paper_tables"

echo "== DONE =="
