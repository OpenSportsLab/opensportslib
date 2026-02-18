CONTRIBUTING.md
This guide outlines the workflow and standards for developers looking to extend or maintain the SoccerNetPro library.

1. Development Environment Setup
To begin contributing, set up a local development environment in "editable" mode so your changes are immediately reflected in the package.

Step 1: Clone the Repository
Bash
git clone https://github.com/OpenSportsLab/soccernetpro.git 
cd soccernetpro
Step 2: Create a Virtual Environment
Use Conda to manage dependencies and ensure Python 3.12 compatibility.

Bash
conda create -n SoccerNet python=3.12 pip
conda activate SoccerNet
Step 3: Install in Editable Mode
Install the base package or include optional dependencies for specific tasks like localization:

Bash
# Install core package in editable mode
pip install -e .

# OR for localization support
pip install -e .[localization]
2. Branching and Merging Rules
The project follows a strict branching strategy to maintain stability in the dev and main branches.

Check Status: Verify your current branch is dev using git status.

Sync: Pull the latest changes: git pull.

Create Feature Branch: Create a new branch from dev: git checkout -b <new_feature/fix/bug>.

Submit PR: 
Before raising a PR, merge latest changes of dev branch and resolve conflicts if any.
Then Raise a Pull Request (PR) to merge your branch back into the dev branch.