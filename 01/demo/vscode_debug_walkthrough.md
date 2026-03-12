# VS Code Debugging Walkthrough

Companion guide for `03a_buggy_bmi.py` demonstrating VS Code's debugging toolkit:
- Setting breakpoints
- Using Variables, Watch, and Debug Console panels
- Stepping through code
- Fixing bugs with runtime inspection

## Setup

1. Open `lectures/01/demo/03a_buggy_bmi.py` in VS Code
2. Ensure Python extension is installed and correct interpreter selected
3. Open Debug panel (bug icon in left sidebar or `Cmd/Ctrl+Shift+D`)
4. Click "Run and Debug" → select "Python File"

## Known Bugs

This script has THREE intentional bugs:
1. **BUG 1:** BMI formula error - uses `weight / height` instead of `weight / (height ** 2)`
2. **BUG 2:** NameError - typo `catgory` vs `category` in `get_bmi_category()`
3. **BUG 3:** IndexError - off-by-one loop error in `print_recommendations()`

## Demonstration Flow

### BUG 1: Formula Error - Demonstrate Runtime Inspection

**Set breakpoint:**
- Click in gutter next to line 21: `bmi = weight / height`
- Red dot appears

**Start debugging:**
- Press F5 or click green play button
- Execution pauses at breakpoint

**Use Variables panel (left side):**
- Expand "Locals" section
- See `weight = 70`, `height = 1.75`
- Note: These look reasonable

**Add Watch expression:**
- Click "+" in Watch panel
- Add: `weight / (height ** 2)`
- Shows ~22.9 (correct BMI)

**Step Over (F10):**
- Execute the buggy line
- Variables panel updates: `bmi = 40.0` (wrong!)
- Compare to Watch expression: 40.0 vs 22.9

**Debug Console (bottom):**
- Type: `height ** 2`
- Shows: `3.0625` (correct square)
- Type: `weight / height`
- Shows: `40.0` (what the code actually does)

**Fix the bug:**
- Stop debugger (red square)
- Change line 21 to: `bmi = weight / (height ** 2)`

### BUG 2: NameError - Variable Typo

**Continue execution:**
- Run debugger again (F5)
- Skip through first breakpoint (Continue or F5)
- Execution stops with NameError at line 37

**Inspect the error:**
- Variables panel shows:
  - `catgory = "Normal weight"` (exists)
  - Trying to return `category` (doesn't exist)
- Error message points to line 37: `return category`

**Debug Console investigation:**
- Try: `catgory`
- Shows: `"Normal weight"`
- Try: `category`
- Shows: `NameError`

**Fix the bug:**
- Change line 37 to: `return catgory`
- OR rename all `catgory` to `category` (lines 29-35)

### BUG 3: IndexError - Off-by-One Loop

**Set conditional breakpoint:**
- Right-click gutter on line 52: `for i in range(1, len(category_ids)):`
- Select "Add Breakpoint" → "Conditional Breakpoint"
- Condition: `i == len(category_ids) - 1`
- This pauses only on the last iteration

**Run and observe:**
- Start debugger (F5)
- Pauses on last loop iteration
- Variables panel shows: `i = 3` (for 4 items)

**Add Watch expressions:**
- `len(category_ids)` → shows 4
- `i` → shows 3
- `i + 1` → shows 4 (out of bounds!)
- `recommendations[i+1]` → will throw IndexError

**Debug Console investigation:**
- Type: `range(1, len(category_ids))`
- Shows: `range(1, 4)` - starts at 1, not 0!
- Type: `list(range(1, len(category_ids)))`
- Shows: `[1, 2, 3]` - skips first item!

**Fix the bug:**
- Change line 52 to: `for i in range(len(category_ids)):`
- Change line 53 to: `print(f"{i+1}. {recommendations[i]}")`
  - Use `i+1` for display numbering, `i` for indexing

### Final Verification

**Remove breakpoints:**
- Click red dots to remove them
- Or: "Remove All Breakpoints" in Debug menu

**Run normally:**
- Run script without debugger: `python 03a_buggy_bmi.py`
- Should complete without errors
- Should print all patient data and all 4 recommendations

## Key Debugging Lessons

**VS Code debugging panels are powerful:**
- **Variables panel:** Automatically shows all local and global variables
- **Watch panel:** Track custom expressions that update each step
- **Debug Console:** Execute any Python code while paused (like a REPL)

**When to use each:**
- **Print debugging:** Quick checks, early stage debugging
- **Breakpoints:** Pause at specific lines to inspect state
- **Conditional breakpoints:** Pause only when condition is true (e.g., specific iteration)
- **Watches:** Track changing values across multiple steps
- **Debug Console:** Investigate hypotheses ("what if I try...?")

**Step commands:**
- **Step Over (F10):** Execute current line, don't enter functions
- **Step Into (F11):** Enter function calls to debug them
- **Step Out (Shift+F11):** Finish current function, return to caller
- **Continue (F5):** Run until next breakpoint

**Always restart kernel/clean state after debugging:**
- Restart ensures fixes work from clean slate
- Run All to verify end-to-end success

## Part C: Writing Tests to Lock In Fixes

After debugging and fixing bugs, write tests to ensure they stay fixed. This is how assignments are autograded.

### Why Write Tests?

**Tests prevent regressions:**
- You fix a bug, but later changes might break it again
- Tests catch this immediately
- Assignments use pytest to autograde your work

**Tests document expected behavior:**
- Shows what "correct" means
- Helps others understand your code

### Creating a Test File

**File:** `test_03a_bmi.py` (already created in demo folder)

Tests follow a pattern:
1. Import the functions you want to test
2. Write functions starting with `test_`
3. Use `assert` to check expected behavior
4. Run with `pytest`

### Test Structure Example

```python
def test_bmi_calculation():
    """Test BMI formula is correct."""
    bmi = calculate_bmi(70, 1.75)  # 70 kg, 1.75 m
    assert 22 < bmi < 23, f"Expected BMI ~22.9, got {bmi}"

def test_bmi_category():
    """Test categorization logic."""
    category = get_bmi_category(22.0)
    assert category == "Normal weight"
```

### Running Tests

**From command line:**
```bash
# Run all tests in file
pytest test_03a_bmi.py -v

# Run specific test
pytest test_03a_bmi.py::test_bmi_calculation -v

# Run all tests in directory
pytest -v
```

**In VS Code:**
1. Open Testing panel (beaker icon in sidebar)
2. Click "Configure Python Tests" → select pytest
3. Tests appear in tree view
4. Click play button next to test to run it
5. Green checkmark = pass, red X = fail

### Test Output Example

```
test_03a_bmi.py::TestCalculateBMI::test_normal_weight_calculation PASSED
test_03a_bmi.py::TestCalculateBMI::test_bmi_bounds PASSED
test_03a_bmi.py::TestGetBMICategory::test_normal_weight_category PASSED
test_03a_bmi.py::TestGetBMICategory::test_boundary_values PASSED

==== 4 passed in 0.03s ====
```

### Connection to Assignments

**Assignment autograding works the same way:**
- Your code is in a file (e.g., `assignment.py`)
- Tests are in `.github/tests/test_assignment.py`
- GitHub Actions runs `pytest` when you push
- Green checkmark on GitHub = tests pass = assignment complete

**Example assignment test:**
```python
def test_student_function():
    """Test student's implementation."""
    from assignment import process_data

    result = process_data("test_input.csv")
    assert len(result) == 50
    assert result["bmi"].between(15, 50).all()
```

### Writing Your Own Tests

**Good test practices:**
- Test normal cases (typical inputs)
- Test boundary cases (edge values like 0, empty, very large)
- Test error cases (invalid inputs)
- Use descriptive test names
- One assertion per test (easier to debug failures)

**Example for notebook Demo 3b:**
```python
def test_elderly_filter_after_fix():
    """Test age filtering works with numeric types."""
    import pandas as pd

    patients = pd.DataFrame({
        "age": ["70", "30", "80"],  # String ages
        "name": ["Alice", "Bob", "Carol"]
    })

    # This was BUG 1 - string comparison doesn't work
    # Fix: convert to numeric first
    patients["age"] = pd.to_numeric(patients["age"])
    elderly = patients[patients["age"] > 65]

    assert len(elderly) == 2, "Should find 2 elderly patients"
```

### Running Tests Before Committing

**Best practice workflow:**
1. Fix bug using debugger
2. Write test to verify fix
3. Run test: `pytest test_file.py -v`
4. Commit if tests pass
5. Push to GitHub (triggers autograding)

**Quick pre-commit check:**
```bash
# Run all quality checks
ruff check .
pytest -q
git commit -m "fix: correct BMI formula"
```

### Summary

**Debugging workflow with tests:**
1. **Bug appears** → Use debugger to find root cause
2. **Fix bug** → Change code based on inspection
3. **Write test** → Lock in the fix with pytest
4. **Run test** → Verify fix works
5. **Commit** → Tests prevent future regressions

This is the same workflow used for assignment autograding:
- You write/fix code
- GitHub Actions runs pytest
- Tests pass = assignment complete
