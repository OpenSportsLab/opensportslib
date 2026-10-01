# CONTRIBUTING.md
This guide outlines the workflow and standards for developers looking to extend or maintain the OpenSportsLib library.

## AI Agent Contributions
For AI-agent driven development, follow `AGENTS.md` in the repository root.

## 1. Development Environment Setup
To begin contributing, set up a local development environment in "editable" mode so your changes are immediately reflected in the package.

#### Step 1: Clone the Repository
```bash
git clone https://github.com/OpenSportsLab/opensportslib.git 
cd opensportslib
```
#### Step 2: Create a Virtual Environment
Use Conda to manage dependencies and ensure Python 3.12 compatibility.
```bash
conda create -n osl python=3.12 pip
conda activate osl
```
#### Step 3: Install in Editable Mode
Install the base package or include optional dependencies for specific tasks like localization:
```bash
# Install core package in editable mode
pip install -e .
```

#### Step 4: Setup Environment (PyTorch, CUDA aware & Optional Dependencies)
```bash
# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali
```

## 2. Branching and Merging - Daily workflow for developers

#### Branches
*main* → stable, production-ready
*dev* → active development integration branch
*dev-<name>* → developer personal branch
*feature-<name>* → new features
*fix-<name>* → bug fixes

#### Rules
- ❌ Never push directly to `main`
- ❌ Never commit directly to `dev`
- ✅ Always create a feature branch from `dev`
- ✅ Always use Pull Requests
- ✅ PRs must target `dev`, NOT `main`

### 1. Sync Repo
Verify your current branch is `dev` and pull the latest changes before starting work.
```bash
git checkout dev
git pull origin dev
```

### 2. Create Feature Branch
Create a new branch from the `dev` source using descriptive naming conventions.
```bash
git checkout -b feature-<feature_name>
```
Naming Examples:
- *feature-model*
- *feature-new-dataset*

### 3. Work Locally
Commit your work often using the following commit style guidelines:

- *feat:* New feature
- *fix:* Bug fix
- *refactor:* Code cleanup
- *docs:* Documentation update

Example commit:
```bash
git add . 
or 
git add -u

git commit -m "feat: add model registry"
```

### 4. Push Branch (just once)
Push your feature branch to the remote repository.
```bash
git push origin feature/your-feature-name
```

### 5. Open Pull Request (PR) → dev
Raise a Pull Request (PR) to merge your branch back into the `dev` branch.

Package versions are managed by GitHub Actions. Do not change the root
`pyproject.toml` package version or the OpenSportsLib dependency pin in
`server/pyproject.toml` in a PR to `dev`; the required Version Integrity check
will reject those edits. Maintainers should follow the
[release-management guide](docs/developer/release-management.md) for feature
release-line preparation and stable releases.

### Contributor License Agreement

Every distinct GitHub-linked commit author in a PR targeting `dev` must accept
the [OpenSportsLab Individual Contributor License Agreement](.github/CLA.md).
When the `CLA check` reports a missing signature, each listed author must add
this exact comment to the PR:

> I have read the OpenSportsLab Individual Contributor License Agreement and I hereby sign it.

The check records the GitHub account, agreement version, document hash, and
acceptance timestamp in the project signing ledger. It blocks merging until all
commit authors are recorded. Use the [OpenSportsLab GitHub organization](https://github.com/OpenSportsLab)
for CLA questions, notices, or revocation requests. Contributions from
unlinked commit identities must be amended to use a GitHub-linked author before
they can pass the check.

### Maintainer CLA setup

Before making the `CLA check` required:

1. Create a GitHub issue named **CLA Signature Ledger**.
2. Add that issue's number as `ledger_issue` in
   [`.github/cla.yml`](.github/cla.yml).
3. Deploy the CLA workflow to the `main` branch. This lets GitHub process
   CLA-signing comments on pull requests.
4. After setting up the CLA workflow, create a test pull request. Wait for the
   `CLA check` to appear.
5. In the `dev` branch settings, make the `CLA check` mandatory before a pull
   request can be merged.
6. Do not allow normal direct pushes to `dev` or allow people to skip required
   checks. Only approved repository administrators may bypass these rules when
   necessary.
7. The automated `github-actions[bot]` account may skip these checks only when
   it updates version information.

✅ PR Checklist:
- [ ] Tests Pass: All existing logic remains functional.
- [ ] Runs on GPU: Code is compatible with CUDA environments.
- [ ] Config Works: YAML configurations resolve correctly.
- [ ] Docs Updated: Relevant documentation reflects your changes.
