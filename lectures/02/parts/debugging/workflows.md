# Debugging Workflows and Strategies 🧭

---

## Breakpoints: Pausing Your Program

- A **breakpoint** stops your program at a specific line
- Lets you **inspect variables and flow** at that moment
- Set breakpoints in:
  - Code: `breakpoint()` or `pdb.set_trace()`
  - IDE: click next to line number

<!--
Breakpoints help you freeze time and examine program state.
-->

---

## Stepping Through Code

- **Step over (`n`)**: run current line, move to next
- **Step into (`s`)**: enter a function call
- **Step out**: finish current function, return to caller
- Use these to **follow the program's path** and see where things go wrong

<!--
Stepping helps trace the exact sequence of execution.
-->

---

## Inspecting Variables

- Use `print()` or debugger commands (`p var`) to see variable values
- Check if values **match your expectations**
- Look for **unexpected `None`, empty lists, wrong types**

<!--
Inspecting variables reveals incorrect data causing bugs.
-->

---

## Conditional Breakpoints

- Break only when a **certain condition is true**
- Example: stop only if `age < 0`

In code:

```python
if age < 0:
    breakpoint()
```

In IDE:

- Right-click breakpoint, add condition `age < 0`

<!--
Conditional breakpoints save time by stopping only on suspicious cases.
-->

---

## Exploring the Call Stack

- The **call stack** shows how you got to the current line
- Helps trace the **sequence of function calls**
- Useful for understanding **context** of the bug

<!--
Call stack exploration reveals the path leading to the error.
-->

---

## Using Test Cases to Reproduce Bugs

- Create **small, repeatable examples** that trigger the bug
- Automate with `assert` statements or test frameworks
- Ensures bug is fixed and **doesn't come back**

Example:

```python
def test_bmi():
    assert calculate_bmi(70, 1.75) > 0
```

<!--
Reproducible tests make debugging and future maintenance easier.
-->

---

# Next: Debugging Examples, Challenges, and Practice 🧩