# Debugging Tools and Techniques 🛠️🐍

---

## In-Code Debugging: Your First Line of Defense

### Print Statements

- The **simplest** way to see what's happening
- Print variable values, types, and checkpoints
- Example:

```python
print("Patient age:", age)
print("Type of diagnosis:", type(diagnosis))
```

<!--
Print statements help trace program flow and catch unexpected values early.
-->

---

### Logging

- Like print, but more **organized and flexible**
- Supports levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Can write to files, filter messages
- Example:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Loading patient data")
logging.debug("Patient ID: %s", patient_id)
```

<!--
Use INFO for general updates, DEBUG for detailed tracing, and ERROR for problems.
-->

---

### Conditional Debug Code

- Add checks that only run when debugging
- Example:

```python
DEBUG = True
if DEBUG:
    print("Debug info:", some_variable)
```

<!--
Conditional debug code helps avoid cluttering output during normal runs.
-->

---

### Using `break` and `exit`

- `break` stops a loop early
- `exit()` stops the whole program
- Useful to **stop execution** when something goes wrong

```python
if patient_age < 0:
    print("Invalid age!")
    exit()
```

<!--
Early exits prevent bad data from causing more errors downstream.
-->

---

## Console Debugging with `pdb` 🐞

`pdb` is Python's built-in **interactive debugger**.

- Lets you **pause** your program and inspect variables
- Step through code **line by line**

### How to Use

- Insert `breakpoint()` or `import pdb; pdb.set_trace()` in your code
- Run your script
- When it hits the breakpoint, you'll see `(Pdb)` prompt

### Common Commands

| Command | What it does |
|---------|--------------|
| `n`     | Next line (step over) |
| `s`     | Step into function |
| `c`     | Continue until next breakpoint |
| `q`     | Quit debugger |
| `p var` | Print variable `var` |
| `l`     | List code around current line |

<!--
`pdb` helps you explore program state interactively, making bugs easier to find.
-->

---

## Debugging in VS Code 🖥️

VS Code has a **graphical debugger** that makes things easier:

- Set breakpoints by clicking next to line numbers
- Run in **debug mode** to pause at breakpoints
- Step through code, inspect variables, watch expressions
- View call stack to see how you got there
- Configure debug settings in `.vscode/launch.json`

<!--
Graphical debugging is beginner-friendly and powerful, especially for complex bugs.
-->

---

## Advanced Tools (Just a Taste)

- **ipdb**: `pdb` with IPython features (tab completion, syntax highlighting)
- **pudb**: Full-screen console debugger with UI
- **Remote Debugging**: Attach debugger to code running elsewhere (e.g., server)
- **Profilers**: Find slow parts of code (`cProfile`, `snakeviz`)
- **Linters**: Catch errors and style issues before running code (`pylint`, `flake8`)

<!--
Advanced tools can save time and catch subtle bugs, but start with basics first.
-->

---

## Demo Break #3: Using Debuggers (15 minutes)

**Goal:** Practice using `pdb` and VS Code debugger.

### Steps:

1. Insert `breakpoint()` in a buggy script.
2. Run the script, explore with `pdb` commands.
3. Set breakpoints in VS Code, run in debug mode.
4. Step through code, inspect variables.
5. Fix the bug and rerun.

**Expected Outcome:** Students can pause code, inspect state, and step through execution confidently.

<!--
Debuggers let you "freeze time" and explore your program, making invisible bugs visible.
-->

---

# Next: Simplifying Failures and Understanding Dependencies 🔬