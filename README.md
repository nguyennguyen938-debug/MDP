# TestGenRL Artifact

Artifact for "MDP-Guided Reinforcement Learning for Cost-Aware LLM-Based Test Generation".

## Contents

- `testgenrl/`: MDP, action, planner, reward, and logging framework code.
- `scripts/`: dataset construction, experiment, held-out evaluation, and aggregation scripts.
- `tests/`: regression tests (see **Tests** for the in-scope subset).
- `configs/`: budget and horizon configuration files.
- `data/`: subject manifests and local benchmark inputs.
- `models/full_dataset/`: trained planner model metadata (JSON).
- `results/full_dataset/`: replay-only event logs from the large-scale replay corpus.
- `dqn_eval/`: the held-out evaluation that backs the paper's reported numbers (subjects,
  trained Double-DQN models, per-profile result logs, and `paper_tables_*`).
- `paper_tables/full_dataset/`: derived CSV and LaTeX tables from the replay corpus.
- `pyproject.toml`, `package.json`: environment metadata.
- `docs/project-README.md`: source project README snapshot.

## Tests

The in-scope TestGenRL test suite passes from the artifact root:

```bash
python3 -m pytest tests/test_core.py tests/test_exp_scripts.py tests/test_integration.py tests/test_review.py -q
# -> 37 passed
```

`tests/test_proposal1_eval.py`, `tests/test_sagetransval_full.py`, and `tests/test_extensions.py`
belong to a separate `sagetransval` extension package that is **out of scope** for this artifact and
is not bundled; running the full `pytest tests` command will fail to collect those modules. Use the
command above for the submitted TestGenRL artifact.

## Reproduction

### Tier 1 (included): held-out paper numbers, replay/audit, no credentials

The headline results (held-out test split, conservative Double-DQN, 3 seeds) are reproduced from the
included replay caches and logs under `dqn_eval/`. To re-aggregate the large-scale replay corpus into
the derived CSV/LaTeX tables:

```bash
python3 scripts/aggregate_exp.py \
  --results results/full_dataset \
  --out paper_tables/full_dataset \
  --figures paper_tables/full_dataset/figures
```

The manuscript's reported tables and figures (`results_core_tables.tex`, `results_diagnostics_tables.tex`, `appendix_secondary_results.tex`, `results_dashboard.tex`) come from the **held-out conservative-DDQN evaluation** in `dqn_eval/`.

The statistics added for the reviews are recomputed from the same `dqn_eval/` logs:

```bash
python3 scripts/review_numbers.py            # defaults to dqn_eval/
```

It prints the LinUCB rows (failure-found rate, coverage, return, and boundary/fuzz/LLM action
shares per profile), the failure-found counts with 95% Wilson intervals, exact McNemar p-values of
each Double-DQN seed against Greedy, the fixed schedule and the bandit (per profile, and pooled over
stress and action-balanced against Greedy), and the CQL-weight table for weights 0, 0.3, 0.5, 1.0
and 3.0. `tests/test_review.py` checks the Wilson and McNemar helpers and the LinUCB planner.

### Tier 2 (optional): live-cache rebuild, requires provider credentials

A full live rerun rebuilds provider caches and retrains/evaluates per provider-budget-horizon cell.
It **requires fresh LLM provider credentials** and is **not** runnable from the staged package alone,
because the live `.llm_cache/` directories are omitted (see **Omitted Live Caches**). The held-out
evaluation scripts (`scripts/dqn_eval_run.sh`, `dqn_eval_profiles.sh`, `dqn_eval_rq3.sh`) read a
`GEMINI_API_KEY`/`OPENAI_API_KEY` and populate a local cache on first run.

## LinUCB Baseline

The `linucb` planner (`LinUCBPlanner` in `testgenrl/planners.py`) is disjoint LinUCB (Li et al.,
2010): a ridge regression per action on the 24-dimensional state plus a bias, scored as the
predicted reward plus an exploration bonus `alpha * sqrt(x^T A^-1 x)` with `alpha = 1.0`. `A^-1` is
updated with Sherman-Morrison rank-1 steps, so it needs no linear-algebra dependency. STOP is never
explored and keeps a fixed score of 0. Its logs are under `dqn_eval/results/linucb/`,
`dqn_eval/results_stress/linucb/` and `dqn_eval/results_action_balanced/linucb/`, and replaying it
needs no credentials, for example on the stress profile:

```bash
python3 -m testgenrl.cli run --planner linucb --subjects dqn_eval/subjects_stress \
  --budget dqn_eval/configs/budget.yaml --llm-provider gemini --llm-model gemini-2.5-flash-lite \
  --cache-mode replay --cache-dir dqn_eval/.llm_cache --out <out_dir>
```

## Learned Planner: Masked Double-DQN (conservative, offline)

The `dqn` planner is a trained, state-conditioned masked Double-DQN (`testgenrl/dqn.py`):
an MLP `24->64->64->8` with a separate target network, masked Double-DQN target
(`y = r + (1-done)*gamma*Q_target(s', argmax_{valid} Q_online(s'))`), and experience replay over
the logged transitions. Inference is dependency-light (weights serialize to JSON).

Training is **offline** (it learns from logged baseline trajectories), so it is prone to
overestimating the value of actions that are under-represented in the buffer. We therefore train
with a **Conservative Q-Learning (CQL)** penalty by default (`--cql-alpha 0.5`), which pushes down
the value of under-covered actions. Empirically this removes the planner's over-exploration on easy
regimes (full-pilot coverage 68.2%->97.3%, recovering the boundary-test optimum) **while keeping its
adaptation on adversarial profiles** (stress/action-balanced failure-found rate held at 66.7%/55.6%, mean of 3 seeds,
vs Greedy's 25.0%/16.7%). `--cql-alpha 0` recovers the non-conservative learner; `>=1.0`
over-suppresses exploration. See the evaluation section of the paper for details on the diagnosis, ablation, and related work.

```bash
# train (conservative offline Double-DQN is the default)
python3 -m testgenrl.cli train --runs <runs_dir> --out models/dqn.json
# ablations:  --cql-alpha 0  (vanilla, over-explores)   --cql-alpha 1.0 (over-conservative)
```

## Omitted Live Caches

Live provider cache directories under `runs/full_dataset/**/.llm_cache/` were deliberately not copied:
they are large and can contain live-provider artifacts. The submitted artifact is intended for
replay/audit from the included event logs (`results/full_dataset/`, `dqn_eval/`).
