# CONTRIBUTING.md
This guide outlines the workflow and standards for developers looking to extend or maintain the OpenSportsLib library.

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

# OR for localization support
pip install -e .[localization]
 
# OR for tracking support
pip install -e .[tracking]
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

✅ PR Checklist:
- [ ] Tests Pass: All existing logic remains functional.
- [ ] Runs on GPU: Code is compatible with CUDA environments.
- [ ] Config Works: YAML configurations resolve correctly.
- [ ] Docs Updated: Relevant documentation reflects your changes.
