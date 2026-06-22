#!/usr/bin/env bash
# RQ2 refresh: evaluate the held-out-trained neural Double-DQN (and baselines) on the
# adversarial stress and action_balanced profiles (cross-profile generalization;
# the DQN is trained only on full-pilot train-split logs from dqn_eval_run.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export GEMINI_API_KEY="$(tr -d '\r\n' < "$HOME/Desktop/keys/gemini.md")"

PROVIDER=gemini; MODEL=gemini-2.5-flash-lite; EXP=dqn_eval; PY="python3 -m testgenrl.cli"
BUD="$EXP/configs/budget.yaml"; CACHE="$EXP/.llm_cache"; DQN_SEEDS="0 1 2"; RAND_SEEDS="0 1 2 3 4"

for prof in stress action_balanced; do
  SUBJ="$EXP/subjects_$prof"
  echo "== prepare $prof subjects =="
  $PY prepare --dataset "$prof" --out "$SUBJ"
  echo "== live cache population for $prof (fixed planner) =="
  $PY run --planner fixed --subjects "$SUBJ" --budget "$BUD" \
    --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode live --cache-dir "$CACHE" --out "$EXP/cache_$prof"
  echo "== replay-eval baselines + random + DQN(3 seeds) on $prof =="
  for p in existing fixed greedy bandit; do
    $PY run --planner "$p" --subjects "$SUBJ" --budget "$BUD" \
      --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --out "$EXP/results_$prof"
  done
  for s in $RAND_SEEDS; do
    $PY run --planner random --subjects "$SUBJ" --budget "$BUD" \
      --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --seed "$s" --out "$EXP/results_$prof"
  done
  for s in $DQN_SEEDS; do
    $PY eval --model "$EXP/models/dqn_s$s.json" --subjects "$SUBJ" --budget "$BUD" \
      --llm-provider "$PROVIDER" --llm-model "$MODEL" --cache-mode replay --cache-dir "$CACHE" --seed "$s" --out "$EXP/results_$prof"
  done
  $PY report --runs "$EXP/results_$prof" --out "$EXP/paper_tables_$prof"
done
echo "== DONE profiles =="
