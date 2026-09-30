# Release Management

OpenSportsLib release metadata is managed by GitHub Actions. Contributors must
not edit the root package version or the inference server's OpenSportsLib
dependency pin in a pull request to `dev`; the required **Version Integrity**
check rejects those changes.

## Version policy

GitHub release tags use `v<major>.<minor>.<patch>`, while PyPI package versions
omit the leading `v`.

| Version | Meaning |
| --- | --- |
| `v1.0.0` | First community-verified, stable public API release. |
| `v0.<minor>.0` | Feature release. Before 1.0, documented breaking public-API changes also use a minor release. |
| `v0.<minor>.<patch>` | Backward-compatible bug-fix release. |
| `X.Y.Z.devN` | Development prerelease published from `dev`. |

## Automated workflows

| Workflow | Trigger | Result |
| --- | --- | --- |
| **CI Fast Tests** | Called by pull-request and branch workflows | Installs the package and runs `bash scripts/run_tests.sh`. |
| **CLA automation** | PRs to `dev` and PR comments | Checks that every GitHub-linked commit author accepted the CLA. |
| **Version Integrity** | PRs to `dev` that touch package metadata | Rejects contributor changes to managed version fields. |
| **Deploy Docs** | Documentation PRs; pushes to `main` | Strictly builds the documentation, then deploys it after a `main` push. |
| **Auto Pre-release Publish to PyPI** | Pushes to `dev` | Tests, increments `.devN`, synchronizes metadata, and publishes a prerelease. |
| **Publish Stable Release to PyPI** | Published GitHub Release | Validates, builds, and publishes the tagged stable release; updates `main` and advances `dev`. |
| **Prepare Development Release Line** | Maintainer workflow dispatch | Starts a chosen newer feature-release line at `X.Y.Z.dev0`. |

## Release procedure

1. For a feature release, a maintainer opens **Actions → Prepare Development
   Release Line**, enters a tag such as `v0.4.0`, and runs it. The workflow
   validates the target is newer than the current line and commits
   `0.4.0.dev0` to `dev` as `github-actions[bot]`.
2. Contributors merge normal changes through PRs to `dev`. Each `dev` push
   runs tests and publishes the next development version such as
   `0.4.0.dev1`, `0.4.0.dev2`, and so on.
3. During a release freeze, promote the prepared `dev` commit to `main`.
   Create and publish GitHub Release `vX.Y.Z` from that current `main` commit.
4. Stable-release automation requires the tag source to be `X.Y.Z.devN` with
   synchronized metadata, builds package `X.Y.Z`, and publishes it to PyPI.
   It then records `X.Y.Z` on `main`.
5. Finally, the same workflow verifies that `dev` is still on the released
   line and advances it to `X.Y.(Z+1).dev0`. For example, publishing `v0.3.1`
   automatically changes `dev` to `0.3.2.dev0`.

If `dev` was already moved to a different release line, stable-release
automation fails before modifying it. Resolve that release-line conflict rather
than overwriting version metadata.

## Repository settings

Protect `dev` by requiring pull requests, CI, CLA check, and **Version
Integrity**. Do not allow people to push directly to `dev` or `main`; permit
only the `github-actions[bot]` version-only commits required by these workflows.
The PyPI token remains configured as the `PYPI_API_TOKEN` repository secret.
