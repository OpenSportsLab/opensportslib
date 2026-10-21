# Training launchers

For tennis skeleton/graph classification, first create the stroke clips with
`tools/convert/tennis_to_classification.py`, then run:

```bash
python tools/train/train_tennis_skeleton.py --check-only
python tools/train/train_tennis_skeleton.py --epochs 20 --batch-size 2
```

The launcher keeps pose extraction inside the existing classification model and
does not start training during conversion. See the
[tennis conversion and training guide](../../docs/tools/tennis-classification.md)
for source paths, match splits, optional exact-clip caching and DALI.

## Five sequential training runs

The [Bash script](weekend_train.sh) contains one section per algorithm and
dataset. It runs the full configurations in this order: GAR tracking GIN,
XFoul MViTv2-S, GAR frames VideoMAEv2-Base, tracking action spotting
GraphConvSeq, and video action spotting RNY008-GSM.

From the repository root, activate your OpenSportsLib environment and run:

```bash
hf auth login
tmux new -s osl-weekend
bash tools/train/weekend_train.sh
```

Detach with `Ctrl-B`, then `D`; reconnect with `tmux attach -t osl-weekend`.
The script stops at the first failed command. To continue at a specific section,
set `START_AT` to its number. For example, after the first run has completed:

```bash
set -o pipefail
START_AT=2 bash tools/train/weekend_train.sh 2>&1 | tee -a weekend_runs/weekend.log
```

This skips GAR tracking, starts with XFoul, displays output in the terminal,
and appends stdout and stderr to `weekend_runs/weekend.log`. Existing downloaded
split manifests are reused. It downloads the XFoul and GAR
frames train, valid, and test splits under `/home/giancos/OSLdata`, creates
their local YAML configs in `weekend_runs`, and lets the two HF spotting
configs and GAR tracking config stage their own selected data. Set
`OSL_DATA_ROOT` before launching if you want another dataset directory.

The full 100-epoch video spotting run was measured at about 10 days on this
machine, so the sequence can extend past five nights. The script does not
shorten any training settings.
