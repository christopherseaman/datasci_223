# Demo 2: Interactive Debugging with `pdb` 🐞

## Goal

Use interactive debuggers to pause code, inspect variables, and step through execution.

## Setup

- Provided `demo2-buggy-script.py` with logic errors
- Insert `breakpoint()` statements in the code where you want to pause

## Tasks

1. **Insert `breakpoint()`** near the suspected bug.
2. **Run the script with pdb:**
   ```bash
   python -m pdb demo2-buggy-script.py
   ```
3. **Use `pdb` commands:**
   - `n` (next line)
   - `s` (step into function)
   - `p var` (print variable)
   - `c` (continue)
4. **Inspect variables** and program flow.
5. **Fix the bug.** Rerun to confirm.

## Expected Outcomes

- Students can pause code execution.
- Students can inspect and modify variables interactively.
- Students can step through code to find bugs.

## Notes

<!--
Interactive debugging provides powerful tools to understand complex bugs beyond print statements.
-->