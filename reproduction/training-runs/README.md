# Training evidence

This directory keeps training evidence in Git for model reproduction and
regression comparisons. Live checkpoints and predictions stay under `outputs/`.
Nothing is pushed automatically.

Each paper has `runs/<experiment>/seed-<seed>/` containing the resolved recipe,
implementation hashes, package and GPU versions (`run.json`), dataset manifest
hashes, preflight checks, epoch losses and validation metrics (`history.json`),
latest progress and final test results including the selected checkpoint hash.
These files are mirrored every 30 seconds and at process exit.

`sessions/<UTC timestamp>-<PID>/console.log` preserves stdout and stderr without
truncating earlier sessions. `session.json` records the exact command, exit status,
Git revision, working-tree status and determinism environment settings.
`working-tree.patch` captures tracked changes at launch; untracked files are
listed in session metadata and must also be included when committing the work.
Historical logs imported before the recorder was added are under `legacy/`;
their original launch metadata was not recorded by this tool.

Launch a campaign from the repository root, using the training environment:

```bash
python tools/reproduction/record_training.py \
  --source outputs/tdeed/campaign \
  --archive reproduction/training-runs/tdeed -- \
  python -u tools/reproduction/tdeed.py campaign --resume --verify-hashes \
    --stage main --skip-blocked \
    --priority SoccerNet_small --priority SoccerNet_big \
    --priority SoccerNetBall_challenge1 --priority SoccerNetBall_challenge2 \
    --output outputs/tdeed/campaign
```

Use `nohup` or a service to survive terminal exits. Only one recorder can use an
output campaign at a time. A live session record is not proof its process is
still alive after a host crash; check the process and progress timestamps.
A zero command exit code does not establish paper reproduction: inspect the
ledger for blocked or failed experiments and compare final metrics with targets.

For regression checks, match recipe, implementation and manifest hashes first,
then compare histories and checkpoint hashes for deterministic reruns on the same
stack. For changed hardware or software, compare validation/test metrics with
explicit tolerances and document the change. These artifacts provide baselines;
they do not replace running the regression checks.
