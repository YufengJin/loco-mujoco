# 02-benchmark-generator — 2026-05-03T07:56:00+02:00

**Task:** Benchmark-generator pipeline — RL category, dispatcher mode (auto mode, env-generator just finished cleanly)
**Commit:** b5face8

## Steps

| Step | Tool / Script | Result |
|------|--------------|--------|
| S0 prereq | check `docker/.classification == benchmark`, container `loco-mujoco-headless` Up | ok |
| S1 markdown | reused env-generator probe.json (54 markdown files indexed) | no re-probe needed |
| S2 classify | classifier evidence persisted from prior run (aux 8/8) | RL / dispatcher |
| S3 scripts | `scripts/{run_demo,run_eval,train_rl_policy}.py` already conformant | reused, no edit |
| S4 docker | dispatcher mode → no extra pip install needed; sb3 NOT installed | docker/ untouched |
| S5 L1 | `python -c "import loco_mujoco"` | PASS — `1.1.0` |
| S5 L2 | `RLFactory.make('UnitreeH1') + reset + 10 step + assert finite reward` | PASS — `action_dim=19, total_reward_10=0.2754` |
| S5 L3_RL | `python scripts/run_demo.py --env_name UnitreeH1 --n-steps 20` | PASS — `total_reward=1.429, runtime=4.03s` |
| S5.5 spec | RL category → skip benchmark-spec.json capture | skipped per contract |
| S6 receipts | `history.md` regenerated; `benchmark.md` already current | English-only |

## Notes

- L2 needed loco-mujoco-specific API: `env.reset()` returns ndarray (not 2-tuple), `env.sample_action_space(key)` requires JAX PRNGKey, action_dim lives on `env.info.action_space.shape[0]` (not `env.action_space`).
- env-generator tier1 `torch_cuda=fail` is the documented JAX-only false-positive — explicitly noted in history.md capabilities table; do NOT count as regression.
- `diagnostics_applied`: empty. No env edits required.
- IsaacGym short-circuit: N/A (is_isaacgym=false).
- benchmark-spec.json: NOT generated (RL benchmarks are spec-capture-exempt by skill contract).
