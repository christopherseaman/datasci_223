# Demo 1: Setup and Environment Verification (~30 min)

This demo walks through accepting the GitHub Classroom assignment, setting up your development environment, and verifying everything works before writing code.

## Part 1: GitHub Classroom Setup

### Accept the assignment
1. Click the GitHub Classroom invite link (provided in class)
2. Authorize GitHub Classroom if this is your first time
3. Accept the assignment—this creates a private repo for you

### Enable GitHub Education benefits (if not already done)
- Visit [GitHub Education Pack](https://education.github.com/pack)
- Sign up with your UCSF email
- Benefits include: Copilot, advanced Codespaces hours, free hosting, etc.

## Part 2: Environment Options

Choose **one** workflow and stick with it throughout the course:

### Option A: GitHub Codespaces (recommended for consistency)
**Pros:** Same environment for everyone, nothing to install locally, works on any device
**Cons:** Requires internet, limited free hours (but plenty with Education pack)

1. From your assignment repo page, click **Code** → **Codespaces** → **Create codespace on main**
2. Wait for container to build (uses `.devcontainer` if provided)
3. Terminal opens automatically—you're ready to code

### Option B: Local development (recommended for performance/PHI work)
**Pros:** Faster, works offline, keeps PHI local
**Cons:** Environment differences across machines, more setup

**Mac/Linux:**
```bash
git clone <your-repo-url>
cd <repo-name>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows users:** Consider [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for a Linux environment that matches Codespaces.

## Part 3: Verify Your Setup

### Test 1: Python and packages
```bash
python --version          # Should be 3.11 or 3.12
pip list                  # Check pandas, jupyter, pytest installed
```

### Test 2: Jupyter notebooks
```bash
jupyter notebook         # Should open browser
# OR in VS Code: Open a .ipynb file, select kernel, run a cell
```

### Test 3: Run All on starter notebook
1. Open `starter.ipynb` (or similar provided notebook)
2. Click **Run All** (or **Restart Kernel and Run All**)
3. All cells should execute without errors
4. If errors appear, check requirements.txt installation

### Test 4: Git workflow
```bash
git status                              # Should show clean or untracked files
echo "# Test" > test.md                 # Create test file
git add test.md
git commit -m "test: verify git works"
git push
```

Check your GitHub repo—the commit should appear.

## Part 4: VS Code Essentials

Install these extensions if using local VS Code:
- **Python** (Microsoft): Linting, debugging, IntelliSense
- **Jupyter** (Microsoft): Notebook support in VS Code
- **Pylance** (Microsoft): Fast type checking and autocomplete
- **Ruff** (Astral): Fast Python linter and formatter (recommended)

**Recommended settings:**
- Format on save: `File → Preferences → Settings → "format on save"`
- Auto-save: `File → Auto Save`
- Default formatter: Select "Ruff" or "Black" in settings

**Command Palette** (`Cmd+Shift+P` / `Ctrl+Shift+P`): Your best friend
- `Python: Select Interpreter` → choose your `.venv`
- `Format Document` → auto-format code
- `Developer: Reload Window` → restart if things get weird

## Part 5: Code Quality Tools

Test your linter and formatter setup:

```bash
# Check code quality with ruff (fast linter)
ruff check .

# Auto-format code with ruff or black
ruff format .
# OR
black .

# Run tests (if any exist yet)
pytest -q
```

**What these tools do:**
- **ruff**: Catches common bugs, style issues, and unused imports
- **black/ruff format**: Ensures consistent code formatting
- **pytest**: Runs automated tests to verify your code works

You'll use these throughout the course to keep code clean and catch bugs early.

## Part 6: Notebook Hygiene

Practice clearing notebook outputs before committing (prevents committing stale results or secrets):

```bash
# Clear outputs from a notebook
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace starter.ipynb

# Verify outputs are cleared (open and check)
# Then run the notebook fresh
jupyter notebook starter.ipynb
# Click "Restart Kernel and Run All"
```

**Why clear outputs?**
- Prevents committing stale screenshots or old results
- Ensures notebook is "Run All" ready (reproducible)
- Avoids accidentally committing secrets or PHI in output cells
- Keeps git diffs clean (only code changes, not output changes)

**When to clear:**
- Before committing notebooks to git
- After debugging sessions that produce sensitive output
- When outputs are outdated or misleading

**In VS Code:**
- You can also use: "Clear All Outputs" from the notebook toolbar
- Then "Restart Kernel and Run All" to verify clean execution

## Success Checklist

- [ ] Accepted GitHub Classroom assignment
- [ ] Environment set up (Codespaces or local venv)
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] Starter notebook runs without errors
- [ ] Test commit pushed to GitHub
- [ ] VS Code extensions installed (if local)
- [ ] Linter and formatter work (`ruff check .` runs without errors)
- [ ] Can clear notebook outputs and run fresh

## Common Issues

**Import errors even after pip install:**
- VS Code: Check you selected the right Python interpreter (Command Palette → "Python: Select Interpreter")
- Terminal: Make sure venv is activated (you should see `(.venv)` in prompt)

**Jupyter kernel not found:**
- Install ipykernel: `pip install ipykernel`
- Register kernel: `python -m ipykernel install --user --name=.venv`

**Git push rejected:**
- Check you're on the right branch: `git branch`
- Make sure you committed: `git status`
- If first push: `git push -u origin main`

## Next Steps

Once your environment is verified:
1. Clear test files (`git rm test.md && git commit -m "chore: remove test file"`)
2. Keep this setup throughout the course—don't switch environments mid-semester
3. Get comfortable with `ruff` and `black`—run them before committing code
4. **Remember:** Clear notebook outputs before commits (`nbconvert --ClearOutput...`)
5. Proceed to Demo 2 for defensive programming exercises

## Resources

- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [DataSci 217 Lecture 01](https://www.notion.so/1-Python-the-Command-Line-and-VS-Code-271d9fdd1a1a805784e1fe68dc985696) (deep dive on tooling)
