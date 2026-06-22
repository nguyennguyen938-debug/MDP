# TestGenRL Artifact Manifest

## Included From Repository

- `testgenrl/` -> `testgenrl/`
- `scripts/` -> `scripts/`
- `tests/` -> `tests/` (in-scope subset: `test_core`, `test_exp_scripts`, `test_integration`)
- `configs/` -> `configs/`
- `data/` -> `data/`
- `models/full_dataset/` -> `models/full_dataset/`
- `results/full_dataset/` -> `results/full_dataset/`
- `dqn_eval/` -> `dqn_eval/` (held-out evaluation backing the paper's reported numbers)
- `paper_tables/full_dataset/` -> `paper_tables/full_dataset/`

## Excluded

- `keys/`: credentials.
- `.venv/`, `node_modules/`, `__pycache__/`, `.pytest_cache/`: local tooling/cache directories.
- `runs/full_dataset/**/.llm_cache/`: live provider response caches.

## Notes

The artifact preserves final replay-only logs and derived tables. Rebuilding provider caches requires fresh credentials and is outside this staged submission package.
