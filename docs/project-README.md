# TestGenRL

`TestGenRL` is a Python research framework for modeling LLM-based test generation as a budget-constrained MDP. It includes subject loading, state construction, valid-action masking, cached LLM-style executors, event logging, baseline planners, replay-based DQN training, and report generation for cost-aware failure discovery experiments.

Quick smoke run:

```bash
python3 -m testgenrl.cli prepare --dataset toy --out data/subjects
python3 -m testgenrl.cli run --planner fixed --subjects data/subjects --budget configs/budget.yaml --out runs/demo
python3 -m testgenrl.cli train --runs runs/demo --out models/dqn.json
python3 -m testgenrl.cli eval --model models/dqn.json --subjects data/subjects --budget configs/budget.yaml --cache-mode live --out results/demo
python3 -m testgenrl.cli report --runs runs/demo results/demo --out paper_tables
```

The default MVP is Python-only and ships with 20 toy subjects so the full pipeline can run without external benchmark downloads or live LLM calls.

Full two-provider MVP run:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
python3 scripts/full_run.py
```

The full run uses `gpt-4o-mini` and `gemini-2.5-flash-lite`, prepopulates provider-specific caches in live mode, then evaluates from replay cache only.
If the environment variables are absent, TestGenRL falls back to `keys/openai.md` and `keys/gemini.md`.

Experiment tables and plots for the paper:

```bash
python3 scripts/exp_run.py
python3 scripts/aggregate_exp.py --results results/exp --out paper_tables/exp --figures paper_tables/figures
```

For a no-API smoke run, use `python3 scripts/exp_run.py --providers stub` and then aggregate from that result directory.
