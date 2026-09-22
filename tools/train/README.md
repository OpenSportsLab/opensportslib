# Five sequential training runs

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
The script stops at the first failed command. It downloads the XFoul and GAR
frames train, valid, and test splits under `/home/giancos/OSLdata`, creates
their local YAML configs in `weekend_runs`, and lets the two HF spotting
configs and GAR tracking config stage their own selected data. Set
`OSL_DATA_ROOT` before launching if you want another dataset directory.

The full 100-epoch video spotting run was measured at about 10 days on this
machine, so the sequence can extend past five nights. The script does not
shorten any training settings.
