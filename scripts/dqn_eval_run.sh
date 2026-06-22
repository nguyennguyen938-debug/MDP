#!/usr/bin/env bash
# Rigorous neural Double-DQN evaluation with a held-out test split and multiple seeds.
#   - 50 real subjects (full_pilot: humaneval 20 / mbpp 20 / bugsinpy 10)
#   - DQN trained ONLY on train-split logs (no leakage); evaluated on the held-out test split
#   - 3 independent DQN training seeds; random baseline over 5 seeds
#   - Gemini live calls populate a content-addressed cache; evaluation runs in deterministic replay
set -euo pipefail

cd "$(dirname "$0")/.."   # artifact/
export GEMINI_API_KEY="$(tr -d '\r\n' < "$HOME/Desktop/keys/gemini.md")"

PROVIDER=gemini
MODEL=gemini-2.5-flash-lite
EXP=dqn_eval
PY="python3 -m testgenrl.cli"
DQN_SEEDS="0 1 2"
RAND_SEEDS="0 1 2 3 4"

rm -rf "$EXP"
mkdir -p "$EXP/configs"
printf 'tokens: 1000\nruntime_seconds: 30\nfuzz_iterations: 100\nhorizon: 16\n' > "$EXP/configs/budget.yaml"
BUD="$EXP/configs/budget.yaml"; SUBJ="$EXP/subjects"; CACHE="$EXP/.llm_cache"

echo "== prepare 50 real subjects (humaneval/mbpp/bugsinpy) =="
$PY prepare --dataset full_pilot --out "$SUBJ"

echo "== live: TRAIN-split logs (baselines + random) -> train buffers + cache =="
for p in existing fixed greedy bandit; do
  $PY run --planner "$p" --subjects "$SUBJ" --split train --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode live --cache-dir "$CACHE" --out "$EXP/trainlogs"
done
for s in $RAND_SEEDS; do
  $PY run --planner random --subjects "$SUBJ" --split train --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode live --cache-dir "$CACHE" --seed "$s" --out "$EXP/trainlogs"
done

echo "== live: populate cache for TEST subjects (fixed planner; NOT used for training) =="
$PY run --planner fixed --subjects "$SUBJ" --split test --budget "$BUD" \
  --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode live --cache-dir "$CACHE" --out "$EXP/testcache"

echo "== train neural Double-DQN on TRAIN logs only (3 seeds) =="
for s in $DQN_SEEDS; do
  $PY train --runs "$EXP/trainlogs" --out "$EXP/models/dqn_s$s.json" --gamma 0.95 --steps 4000 --seed "$s"
done

echo "== replay-eval on HELD-OUT TEST split: baselines + random + DQN(3 seeds) =="
for p in existing fixed greedy bandit; do
  $PY run --planner "$p" --subjects "$SUBJ" --split test --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --out "$EXP/results"
done
for s in $RAND_SEEDS; do
  $PY run --planner random --subjects "$SUBJ" --split test --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --seed "$s" --out "$EXP/results"
done
for s in $DQN_SEEDS; do
  $PY eval --model "$EXP/models/dqn_s$s.json" --subjects "$SUBJ" --split test --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --seed "$s" --out "$EXP/results"
done

echo "== report =="
$PY report --runs "$EXP/results" --out "$EXP/paper_tables"
echo "== DONE =="
