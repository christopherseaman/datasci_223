# Lecture 01 Demo Guide

Brief walkthrough for all three hands-on demos. Each demo corresponds to a major section in the lecture.

---

## Demo 1: Setup and Environment Verification

**File:** `01_setup_resources.md` (reference doc, not a notebook)
**Lecture section:** Command line quick hits + Workflow: local venv or Codespaces
**Timing:** After initial setup and workflow sections (approximately 1/3 through lecture)

### Goal
Verify students can:
- Accept GitHub Classroom assignment
- Set up local or Codespaces environment
- Install dependencies
- Run a notebook without errors
- Make a test commit

### Walkthrough

1. **Accept Classroom invite:**
   - Share the Classroom link (add to lecture when created)
   - Students click, accept assignment
   - GitHub creates private repo for each student

2. **Environment setup (pick one):**
   - **Codespaces:** Click Code → Codespaces → Create
   - **Local:** Clone repo, `python -m venv .venv`, activate, `pip install -r requirements.txt`

3. **Verify installation:**
   ```bash
   python --version
   pip list | grep pandas
   jupyter notebook  # or open .ipynb in VS Code
   ```

4. **Test notebook:**
   - Open starter.ipynb (from assignment repo)
   - Click Run All
   - All cells should execute without errors

5. **Test git workflow:**
   ```bash
   echo "# Test" > test.md
   git add test.md
   git commit -m "test: verify git works"
   git push
   ```
   - Check GitHub—commit should appear
   - Clean up: `git rm test.md && git commit -m "chore: remove test file"`

6. **Demonstrate notebook hygiene:**
   - Run starter notebook, show outputs appear
   - Clear outputs: `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace starter.ipynb`
   - Open notebook—outputs are gone
   - Run All—outputs regenerated cleanly
   - Explain: Always clear before committing to avoid stale results or secrets

### Success criteria
- [ ] Students can access their assignment repo
- [ ] Dependencies installed without errors
- [ ] Starter notebook runs
- [ ] Test commit appears on GitHub
- [ ] Students can clear notebook outputs and verify clean run

### Common issues
- **Wrong Python interpreter in VS Code:** Command Palette → "Python: Select Interpreter" → choose `.venv`
- **Jupyter kernel not found:** `pip install ipykernel`
- **Git push rejected:** Check you're on the right branch, made a commit

---

## Demo 2: Defensive Programming

**Files:** `02a_brittle_cleaning.md` (starter) → `02b_hardened_cleaning.md` (solution)
**Lecture section:** Defensive programming for data science + Common failure modes
**Timing:** After defensive programming sections (approximately 2/3 through lecture)
**Prep:** Convert markdown to notebooks before class: `jupytext --to notebook 02*.md`

### Goal
Show how defensive programming prevents silent failures:
- Config-driven paths and bounds
- Schema validation
- Logging for observability (without exposing PHI)
- Informative error messages
- Pure functions for testability

### Walkthrough

1. **Run brittle version with clean data:**
   - Open `02a_brittle_cleaning.ipynb`
   - Run All—it works! (Appears fine)

2. **Try with missing column data:**
   - Edit cell 1: Change path to `"data/patient_intake_missing_height.csv"`
   - Run All → `KeyError: 'height_cm'`
   - **Problem:** Cryptic error, no context

3. **Try with out-of-bounds data:**
   - Change path to `"data/patient_intake_bad_values.csv"`
   - Run All → Completes but produces nonsense BMI values (500 kg patient!)
   - **Problem:** Silent failure—wrong results without errors

4. **Compare to hardened version:**
   - Open `02b_hardened_cleaning.ipynb`
   - Show config file `02_config.yaml` with paths, bounds, thresholds
   - Run All with clean data → Works, with logging statements showing progress

5. **Test hardened version with bad data:**
   - Scroll to "Test failure modes" cell
   - Run—see informative error messages:
     ```
     ERROR: Missing required columns: ['height_cm']
     Available columns: ['patient_id', 'weight_kg', 'age', 'sex']
     ```
     ```
     ERROR: Column 'weight_kg' has values outside [30, 250]
     Problem rows:
        patient_id  weight_kg
     2        P003        500
     ```

6. **Show Jupyter magics section:**
   - Demonstrate `%pwd`, `%timeit`, `!ls`, `%whos`
   - Explain these only work in notebooks, not .py scripts
   - Show how `%timeit` helps profile performance

7. **Highlight key changes:**
   - Config file vs hardcoded values
   - Schema validation function
   - Bounds checking with clear error messages
   - Logging throughout execution (note: never log PHI like patient_id, names, dates)
   - Pure functions (no side effects, easier to test)

### Success criteria
- [ ] Students see brittle version fail silently or cryptically
- [ ] Students understand config-driven development
- [ ] Students can add basic validation to their own notebooks

### Discussion points
- "What happens if you skip validation?" → Silent bugs in production, wrong clinical decisions
- "When is logging overkill?" → Never, for data analysis that runs more than once
- "Why pure functions?" → Easier to test and debug, no hidden side effects
- "What about PHI in logs?" → Never log identifiable data (patient_id, MRN, names, DOB)

---

## Demo 3: VS Code Debugging

**Files:** `03a_buggy_bmi.py` (script) and `03b_buggy_analysis.md` (notebook)
**Lecture section:** Debugging in VS Code + Jupyter
**Timing:** Final demonstration section
**Prep:** Convert notebook: `jupytext --to notebook 03b_buggy_analysis.md`

### Goal
Practice using VS Code debugger for both scripts and notebooks:
- Set breakpoints in gutter (scripts) and cells (notebooks)
- Use Variables panel to inspect state automatically
- Use Watch panel to track custom expressions
- Use Debug Console to evaluate code while paused
- Step through code (Step Into vs Step Over)
- Fix bugs and verify with clean restart

### Part A: Script debugging

**File:** `03a_buggy_bmi.py`

1. **Run script to observe failures:**
   ```bash
   python demo/03a_buggy_bmi.py
   ```
   - Wrong BMI calculations (BUG 1: formula error)
   - NameError on line 32 (BUG 2: typo `catgory` vs `category`)
   - Doesn't reach recommendations (crashes first)

2. **Debug BUG 1: Formula error WITH runtime inspection**
   - Open `03a_buggy_bmi.py` in VS Code
   - Set breakpoint on line 18: `bmi = weight / height`
   - Run debugger (Debug icon → Python File)
   - When paused at breakpoint:
     - **Variables panel:** Expand "Locals" → see `weight=70`, `height=1.75`
     - **Add Watch:** Click "+" in Watch panel → add expression `weight / (height ** 2)`
     - **Compare:** Watch shows ~22.9 (correct) vs bmi=40.0 after step
     - **Debug Console:** Type `height ** 2` → see 3.0625
   - Step Over (F10) to execute line
   - **Variables panel updates:** See `bmi=40.0` (wrong!)
   - Fix formula: `bmi = weight / (height ** 2)`
   - Continue (F5) to next bug

3. **Debug BUG 2: NameError**
   - Execution stops at line 32 with NameError
   - Inspect Variables panel: `catgory` exists but function tries to return `category`
   - Fix typo: rename `catgory` → `category` (or vice versa)
   - Restart debugger

4. **Debug BUG 3: IndexError**
   - Set breakpoint inside `print_recommendations()` loop
   - Step Over (F10) through iterations
   - Watch `i` variable: starts at 1 (skips first item)
   - On last iteration, `recommendations[i+1]` → IndexError
   - Fix: `range(len(category_ids))` and `recommendations[i]`

5. **Verify fix:**
   - Remove breakpoints
   - Run script normally—should complete without errors

### Part B: Notebook debugging

**File:** `03b_buggy_analysis.ipynb`

1. **Run notebook to observe issues:**
   - Run All → several cells fail or produce unexpected results

2. **Debug BUG 1: Type mismatch WITH runtime inspection**
   - Find cell with elderly patient filter
   - Click **debug icon** beside the cell (bug/play button)
   - Set breakpoint on `elderly = patients[patients["age"] > 65]`
   - When paused:
     - **Variables panel:** Expand `patients` → click "View Value" to see DataFrame
     - **Watch panel:** Add `patients["age"].dtype` → shows `dtype('O')` (object/string!)
     - **Debug Console:** Type `patients["age"].head()` → see string values
     - **Add Watch:** `(patients["age"] > 65).sum()` → compare before/after fix
   - Notice: dtype is `object` (string), not `int64`—string comparisons behave differently
   - Fix: `patients["age"] = pd.to_numeric(patients["age"])`
   - Verify in Debug Console: `patients["age"].dtype` → now `int64`

3. **Debug BUG 2: Off-by-one loop error WITH inspection**
   - Debug the summary statistics cell
   - Set breakpoint inside the for loop
   - When paused in first iteration:
     - **Variables panel:** See `i=1` (should start at 0!)
     - **Add Watch:** `len(patient_ids)` → see total count
     - **Add Watch:** `i + 1` → see what index will be accessed
     - **Debug Console:** `patient_ids[0]` → see we're skipping first patient
   - Step Over through iterations, watch `i` increment
   - On last iteration: `i=9`, trying to access `iloc[10]` → IndexError!
   - Fix: `range(len(patient_ids))` and remove the `+1` offset

4. **Debug BUG 3: Logic error in risk categorization**
   - Debug the `categorize_risk()` function cell
   - Step through with test BMI values
   - Notice: BMI=35 returns "Very low risk" (should be high!)
   - Fix: Correct the labels or logic

5. **Verify notebook fixes:**
   - Restart Kernel
   - Run All—should complete without errors
   - Check outputs make sense

### Part C: Writing Tests After Debugging

**File:** `test_03a_bmi.py`

After fixing bugs, demonstrate writing tests to lock in fixes:

1. **Show test file structure:**
   - Open `test_03a_bmi.py` in VS Code
   - Explain test naming: functions start with `test_`
   - Show how to import functions to test

2. **Run tests from command line:**
   ```bash
   pytest test_03a_bmi.py -v
   ```
   - All tests should pass after bugs are fixed
   - Show test output with PASSED/FAILED status

3. **Run tests in VS Code:**
   - Open Testing panel (beaker icon)
   - Configure pytest if needed
   - Click play button to run tests
   - Show green checkmarks for passing tests

4. **Connect to assignment autograding:**
   - Explain assignments use same pytest pattern
   - GitHub Actions runs tests when you push
   - Tests pass = assignment complete
   - Show example: `.github/tests/test_*.py`

5. **Demo writing a new test:**
   - Pick a function from fixed code
   - Write a simple test live
   - Run it with pytest
   - Show it passes

### Success criteria
- [ ] Students can set breakpoints in scripts and notebooks
- [ ] Students use Variables panel to inspect state
- [ ] Students understand Step Into vs Step Over
- [ ] All three bugs fixed in both script and notebook
- [ ] Students can run pytest and understand test output
- [ ] Students understand connection to assignment autograding

### Key debugging lessons

**Progressive debugging approach:**
- **Print debugging:** Start here for quick checks (`print(f"{var=}")`)
- **Breakpoints:** When you need to inspect state at specific points
- **Runtime inspection:** Leverage Variables/Watch/Debug Console panels:
  - **Variables panel:** See all variables automatically
  - **Watch expressions:** Add custom calculations that update each step
  - **Debug Console:** Execute any Python code while paused
- **Step through:** Trace execution flow line by line
- **Restart + Run All:** Always verify fixes in clean state

**Runtime inspection is powerful because:**
- No need to modify code with print statements
- Can evaluate complex expressions on-the-fly
- See variables update in real-time as you step
- Works identically for scripts and notebooks
- Can inspect DataFrames interactively (`.shape`, `.dtypes`, `.head()`)

**After debugging, write tests:**
- Tests prevent regressions (bugs coming back)
- Same pattern as assignment autograding
- Run with `pytest test_file.py -v`
- Green checkmarks = all tests pass = assignment complete

---

## Converting Markdown to Notebooks

Before class, convert demo markdown files:

```bash
cd lectures/01/demo

# Install jupytext if needed
pip install jupytext

# Convert demos 2 and 3
jupytext --to notebook 02a_brittle_cleaning.md
jupytext --to notebook 02b_hardened_cleaning.md
jupytext --to notebook 03b_buggy_analysis.md

# Verify notebooks created
ls *.ipynb
```

**Note:** Keep `.md` files as source of truth for git—notebooks can be regenerated.

---

## Teaching Tips

- **Demo 1:** Keep it quick—assume students know git/python basics from prereq
- **Demo 2:** Emphasize "fail fast with context" philosophy—better to crash with info than succeed with wrong results
- **Demo 3:** Let students drive—ask them to predict what breakpoint will reveal before stepping

**Pacing:**
- Demo 1: Setup is critical, don't rush the environment verification
- Demo 2: Can show before/after quickly if time is tight
- Demo 3: Most engaging, allow time for questions about debugger features

**Engagement hooks:**
- "Raise hand if you've ever had code work once then mysteriously break" (before Demo 2)
- "Who has used print debugging?" (before Demo 3)
- "This bug cost healthcare.gov $300M" (when discussing validation importance)
