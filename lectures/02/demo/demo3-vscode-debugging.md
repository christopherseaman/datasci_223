# Demo: Debugging Python with VS Code 🖥️🐞

---

## Goal

Learn to use **VS Code's graphical debugger** to find and fix bugs efficiently.

---

## Script Overview

`buggy_script.py` calculates a patient's BMI, categorizes it, and prints recommendations. It also contains intentional bugs for debugging practice.

---

## Known Bugs in `buggy_script.py`

- **Incorrect BMI formula:** uses `weight / height` instead of `weight / (height ** 2)`
- **Typo:** uses `catgory` instead of `category`, causing a `NameError`
- **Off-by-one error:** causes an `IndexError` when printing list elements

---

## Setup

- Open `buggy_script.py` in VS Code.
- Ensure the **Python extension** is installed.
- Use the **Run and Debug** panel or the **Run > Start Debugging** menu.

---

## Key Debugging Features in VS Code

### 1. Setting Breakpoints

- Click to the **left of a line number** to add a breakpoint.
- The program will **pause** when it reaches this line.

### 2. Running in Debug Mode

- Click the **Run and Debug** button or press `F5`.
- The script will start and pause at breakpoints.

### 3. Stepping Through Code

- **Continue (F5):** Run until next breakpoint.
- **Step Over (F10):** Run current line, move to next.
- **Step Into (F11):** Enter a function call.
- **Step Out (Shift+F11):** Finish current function, return to caller.

### 4. Inspecting Variables

- Hover over variables to see their values.
- Use the **Variables** panel to watch all current variables.
- Add **Watch expressions** to track specific values.

### 5. Viewing the Call Stack

- See the **sequence of function calls** that led to the current line.
- Helps understand program flow and context.

### 6. Using Conditional Breakpoints

- Right-click a breakpoint > **Edit Breakpoint**.
- Add a condition, e.g., `age < 0`.
- The program will pause **only when the condition is true**.

### 7. Debug Console

- Execute Python commands live while paused.
- Inspect or modify variables interactively.

---

## Tasks

1. **Set a breakpoint** inside the `main()` function (e.g., on the line `bmi = calculate_bmi(...)`).
2. **Run in debug mode** and pause execution.
3. **Step into** the `calculate_bmi` function.
   - **Observe:** The BMI value is suspiciously large or small.
   - **Fix:** Change the formula to `weight / (height ** 2)`.
4. **Continue** to the BMI categorization and print statements.
5. When the program crashes with a `NameError`,
   - **Inspect:** The typo `catgory`.
   - **Fix:** Rename it to `category`.
6. **Run again**. When the program crashes with an `IndexError`,
   - **Inspect:** The loop `for i in range(1, len(numbers))` and the line `numbers[i+1]`.
   - **NOTE:** This is a **compound bug**: it causes an `IndexError` on the last iteration **and** skips printing the first element of the list.
   - **Fix:** Change the loop to `range(len(numbers) - 1)` or adjust indexing accordingly.
7. **Use variable inspection, call stack, and stepping** to understand program flow and confirm each fix.
8. **Rerun** to ensure all bugs are fixed.

---

## Expected Outcomes

- Confidently use VS Code's debugger to find and fix bugs.
- Understand how to pause, inspect, and control program execution.
- Use conditional breakpoints to focus on tricky cases.

---

## Notes

<!--
VS Code's graphical debugger makes complex debugging more approachable, especially for beginners.
-->