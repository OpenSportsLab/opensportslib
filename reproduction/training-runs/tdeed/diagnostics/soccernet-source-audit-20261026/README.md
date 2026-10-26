# SoccerNet source comparison — 2026-10-26

All Hugging Face operations were reads/downloads. No remote repository, branch,
dataset or model was changed.

The preferred media repository is `SoccerNet/SoccerNet_raw_HQ`:

| Representation | Immutable revision |
| --- | --- |
| `videos-224p` | `56a8be6be6788ed65cce25f68780d55ff4544aa2` |
| `videos-720p` | `280e9828acb8f99250eefdeb4c0eae9a1f784739` |
| `videos-HQ` | `462c8ed7da5784d606c6173efe93aa90cc15fea1` |

The comparison covered the 1,000 selected media assets in the current manifests:
982 at 224p and 18 existing 720p fallbacks. Official Hub blob hashes match the
cached original-download hashes for 935 assets. All 65 differing 224p assets
were downloaded, SHA-256 verified, and fully decoded sequentially with PyAV.

Of those 65 files, **63 decode to fewer frames than the benchmark expects**:
35 training, 14 validation, and 14 test halves. Shortfalls range from 1 to 996
frames, totalling 5,317 frames. No sequential decoding errors occurred.
The other 935 files were not fully decoded in this audit; their blob comparison
must not be presented as a complete decoding validation of all 1,000 files.

For Napoli–Udinese, 2017-04-15, first half, the official 224p and 720p versions
both sequentially decode to **67,485 frames**, against 67,500 benchmark frames.
Their headers report 67,500, and direct seeks to the apparent final frames can
succeed. Header counts and successful seeks therefore cannot establish native
frame-index alignment or complete decoded frame counts.

The HQ version has an untrimmed lead-in: `video.ini` specifies a 463-second
offset for this half. Boundary checks include that offset. Its full sequential
decode produces 82,002 frames, although the header reports 82,017. The source
contains enough frames to cover the benchmark interval after the lead-in, but
its header also overestimates the decoded frame count by 15. This does not
certify frame-by-frame alignment with the benchmark. Other games' 720p/HQ
versions have not been validated by this audit.

`official-soccernet-inventory.json` retains the pinned official file inventory;
`official-soccernet-comparison.json` records the 1,000 blob comparisons.
`official-soccernet-replacements.json` is the local decode plan;
`official-soccernet-full-decode.json` contains its completed sequential results.
The other-version plans/results retain the sample's source offsets and checks.

Repeat the 65-file local sequential audit after downloading the pinned assets:

```bash
python tools/reproduction/audit_soccernet_media.py \
  --plan reproduction/training-runs/tdeed/diagnostics/soccernet-source-audit-20261026/official-soccernet-replacements.json \
  --output outputs/tdeed/rechecked-official-soccernet.json --workers 4
```

Exit code 2 means a checksum/decode error or a decoded frame shortfall was
found. Original annotations, benchmark bounds and completed-run provenance
are retained. Switching repositories alone does not resolve the mismatch;
frame extraction, indexing and the reference reader's trailing-padding policy
must be validated before rerunning the affected SoccerNet experiments.

Verification: six relevant source/preparation/audit tests pass, including a
real-video fixture with a gap in frame timestamps and a checksum mismatch check.
The filtered runner's global coverage gate fails because only those tests run.
The full offline suite was attempted and interrupted after stalling in existing
tracking tests; no full-suite pass is claimed.
