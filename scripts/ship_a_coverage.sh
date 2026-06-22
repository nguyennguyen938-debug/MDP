#!/usr/bin/env bash
# "Ship A": coverage-aware conservatism. Train the conservative (CQL) Double-DQN on a
# buffer that INCLUDES adversarial-profile coverage (full-pilot + stress + action_balanced
# train splits), so conservatism keeps fuzz/LLM where they are genuinely valuable.
# Held-out test per profile. All log-gen + eval run against the existing populated cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export GEMINI_API_KEY="$(tr -d '\r\n' < "$HOME/Desktop/keys/gemini.md")"
P=gemini; M=gemini-2.5-flash-lite; EXP=dqn_eval; PY="python3 -m testgenrl.cli"
BUD="$EXP/configs/budget.yaml"; CACHE="$EXP/.llm_cache"
subj_dir() {  # map profile -> prepared subjects dir (bash 3.2 compatible)
  case "$1" in
    pilot)  echo "$EXP/subjects" ;;
    stress) echo "$EXP/subjects_stress" ;;
    ab)     echo "$EXP/subjects_action_balanced" ;;
  esac
}

echo "== generate coverage-enriched TRAIN logs (train split of every profile) =="
rm -rf "$EXP/trainlogs_cov"
for prof in pilot stress ab; do
  for p in existing fixed greedy bandit; do
    $PY run --planner "$p" --subjects "$(subj_dir "$prof")" --split train --budget "$BUD" \
      --llm-provider "$P" --llm-model "$M" --cache-mode live --cache-dir "$CACHE" --out "$EXP/trainlogs_cov" >/dev/null
  done
  for s in 0 1 2 3 4; do
    $PY run --planner random --subjects "$(subj_dir "$prof")" --split train --budget "$BUD" \
      --llm-provider "$P" --llm-model "$M" --cache-mode live --cache-dir "$CACHE" --seed "$s" --out "$EXP/trainlogs_cov" >/dev/null
  done
done

echo "== train: vanilla(coverage) and CQL(coverage), 3 seeds each =="
for s in 0 1 2; do
  $PY train --runs "$EXP/trainlogs_cov" --out "$EXP/models/cov_s$s.json"    --gamma 0.95 --steps 4000 --seed "$s" --cql-alpha 0 >/dev/null
  $PY train --runs "$EXP/trainlogs_cov" --out "$EXP/models/covcql_s$s.json" --gamma 0.95 --steps 4000 --seed "$s" --cql-alpha 1 >/dev/null
done

echo "== eval on held-out TEST split per profile =="
for prof in pilot stress ab; do
  for tag in cov covcql; do
    for s in 0 1 2; do
      $PY eval --model "$EXP/models/${tag}_s$s.json" --subjects "$(subj_dir "$prof")" --split test --budget "$BUD" \
        --llm-provider "$P" --llm-model "$M" --cache-mode replay --cache-dir "$CACHE" --seed "$s" \
        --out "$EXP/shipA_${prof}_${tag}" >/dev/null
    done
  done
  $PY run --planner greedy --subjects "$(subj_dir "$prof")" --split test --budget "$BUD" \
    --llm-provider "$P" --llm-model "$M" --cache-mode replay --cache-dir "$CACHE" --out "$EXP/shipA_${prof}_greedy" >/dev/null
done
echo "== DONE ship A coverage experiment =="
