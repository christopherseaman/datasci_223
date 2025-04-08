# Jupyter Notebooks: Your Interactive Coding Playground 🧑‍💻📓

---

## What is Jupyter?

Jupyter notebooks are interactive documents that let you **write and run code, see results immediately, and mix in text, images, and equations**.

- Great for **exploring data**, **trying out code**, and **sharing your work**.
- Widely used in **health data science** for analysis, visualization, and reporting.

<!--
- Jupyter is beginner-friendly and forgiving, making it easy to experiment.
- You can safely try out code and learn by doing.
- Notebooks can be exported as reports or converted into scripts for sharing or reuse.
-->

---

## Magic Commands ✨

Jupyter has special commands starting with `%` or `%%` called **magics**.

- **`%pip install package_name`**: Install Python packages *inside* the notebook.
- **`%timeit some_code`**: Measure how long code takes to run.
- **`%debug`**: Enter interactive debugger after an error.
- **`%run script.py`**: Run an external Python script.
- **`%load script.py`**: Load a script's content into a cell.
- **`%store`**: Save variables between sessions.

Example:

```python
%pip install pandas
```

<!--
- `%pip` allows you to install packages directly inside a notebook.
- `%timeit` helps you measure how long code takes to run.
- `%debug` provides an interactive debugger after errors, covered later in debugging.
-->

---

## Shell Commands in Jupyter 🐚

You can run **Linux shell commands** by prefixing with `!`.

- `!ls` — list files in the current directory
- `!pwd` — print working directory
- `!echo Hello` — print text

Example:

```python
!ls
```

<!--
- Prefixing with `!` lets you run shell commands inside Jupyter, just like in a terminal.
- Useful for checking files, running scripts, or managing data without leaving the notebook.
-->

---

## Best Practices for Notebooks 🧹

- **Keep it clean:** Remove failed code, unnecessary outputs.
- **Use Markdown cells** for explanations, titles, and notes.
- **Restart and run all** before sharing to ensure reproducibility.
- **Export notebooks** as HTML or PDF for reports.
- **Convert to scripts** (`File > Export`) for production code.

<!--
- Use Markdown cells to clearly explain your analysis and results.
- Restarting and running all cells ensures your work is reproducible by others or your future self.
-->

---

## Resources 📚

- [Dataquest Jupyter tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
- [Jupyter documentation](https://jupyter.org/documentation)

---

## Live Demo!

---

# Next: Debugging Python Code 🐛🔍# Debugging Python: Mindset, Common Errors, and Observing Behavior 🐛🔍

---

## Why Debugging Matters

Debugging is **figuring out why your code doesn't work** — a core skill for every programmer.

- **Programming = solving problems + fixing mistakes**
- Bugs are **normal**, not a sign of failure
- Debugging is like **detective work**: gather clues, test theories, find the culprit

<!--
- Bugs are a normal part of programming and learning.
- Debugging requires patience and curiosity to find solutions.
- Like medical diagnosis, debugging involves observing symptoms, testing ideas, and finding the root cause.
-->

---

## The Debugging Mindset 🧠

- **Be systematic:** Change one thing at a time, observe effects
- **Minimize the problem:** Simplify inputs, isolate code
- **Use the scientific method:** Form hypotheses, test, revise
- **Explain it aloud:** Rubber duck debugging 🦆
- **Don't guess wildly:** It wastes time and causes confusion

<!--
- Explaining your thought process aloud can help you spot mistakes.
- Discussing problems with others or even "rubber duck debugging" can clarify your thinking.
- Even experienced programmers debug their code regularly.
-->

---

## Common Python Errors (and How to Fix Them) 🚨

| **Error Type** | **What it Means** | **Common Causes** | **Fix Tips** |
|----------------|-------------------|-------------------|--------------|
| `SyntaxError` | Python can't understand your code | Missing colon, parentheses, quotes | Check punctuation carefully |
| `IndentationError` | Bad indentation | Mixing tabs/spaces, wrong indent level | Use consistent 4 spaces |
| `NameError` | Variable/function not defined | Typos, using before assignment | Check spelling, order |
| `AttributeError` | Object lacks attribute | Typos, wrong object type | Print object type, check docs |
| `FileNotFoundError` | File doesn't exist | Wrong path, file missing | Use `!ls` to check files |
| `IndexError` | List/string index out of range | Off-by-one errors | Print list length, index |
| `ImportError` | Can't find module | Typos, missing install | Check spelling, `%pip install` |
| `TypeError` | Wrong data type used | Adding string + int, bad function args | Print types, use `type()` |
| `ValueError` | Bad value for function | Converting 'abc' to int | Print values before convert |

<!--
- Carefully reading the full error message helps identify the problem.
- The traceback shows where the error occurred in your code.
- Searching error messages online is a common and effective debugging strategy.
-->

---

## Observing Program Behavior 👀

Before fancy tools, **start by watching what your code does**:

### Print Statements

- Print variables at key points
- Check assumptions: "Is this what I expect?"
- Example:

```python
age = "30"
print("age type:", type(age))  # <class 'str'>
age = int(age)
print("converted age:", age)   # 30
```

### Logging (Preview)

- Like print, but more flexible
- Can set levels: DEBUG, INFO, WARNING, ERROR
- We'll cover this more in the tools section

### Using Assertions

- Check that something **must** be true
- If not, program stops with an error
- Example:

```python
bmi = weight / (height ** 2)
assert bmi > 0, "BMI should be positive"
```

### Collecting Failure Data

- What inputs cause the bug?
- What was the program state?
- Can you reproduce it every time?

### Reproducing Bugs Reliably

- Use the **same data and steps** each time
- Simplify inputs to the smallest failing case
- This makes debugging much easier!

<!--
- Adding print statements helps you observe what your code is doing.
- Being able to reliably reproduce a bug is essential for fixing it.
- Simplifying failures to the smallest example makes debugging easier.
-->

---

## Live Demo!
See demo: [`demo1-print-debugging`](lectures/02/demo/demo1-print-debugging.md)

---

# Next: Debugging Tools and Techniques 🛠️# Debugging Tools and Techniques 🛠️🐍

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
- Adding print statements helps you trace how your program runs step by step.
- They reveal unexpected values or logic errors during execution.
- Print debugging is a simple but powerful first tool for understanding code behavior.
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
- Logging provides organized, adjustable output for monitoring your program.
- Use INFO for general updates, DEBUG for detailed tracing, WARNING/ERROR for issues.
- Logging can be saved to files and filtered, making it more flexible than print statements.
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
- Conditional debug code lets you enable or disable extra output easily.
- This keeps your program output clean during normal use.
- You can add debug info without permanently cluttering your code.
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
- Using `exit()` or `break` can stop your program early if something is wrong.
- This prevents bad data or errors from causing bigger problems later.
- Adding early checks improves program safety and clarity.
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
- The `pdb` debugger lets you pause your program and inspect variables interactively.
- You can step through code line by line to understand what happens.
- This makes finding and fixing bugs much easier than guessing.
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
- Graphical debuggers like in VS Code provide a visual way to pause and inspect your code.
- You can set breakpoints, watch variables, and step through execution easily.
- This approach is beginner-friendly and very effective for complex bugs.
-->

---

## Advanced Tools (Just a Taste)

- **ipdb**: `pdb` with IPython features (tab completion, syntax highlighting)
- **pudb**: Full-screen console debugger with UI
- **Remote Debugging**: Attach debugger to code running elsewhere (e.g., server)
- **Profilers**: Find slow parts of code (`cProfile`, `snakeviz`)
- **Linters**: Catch errors and style issues before running code (`pylint`, `flake8`)

<!--
- Advanced debugging tools offer features like better interfaces, remote debugging, and profiling.
- They can save time and catch subtle bugs in larger projects.
- It's best to master basic debugging first before exploring these advanced options.
-->

---

## Live Demo!
See demos: [`demo2-pdb-debugging`](lectures/02/demo/demo2-pdb-debugging.md), [`demo3-vscode-debugging`](lectures/02/demo/demo3-vscode-debugging.md)

---

# Next: Simplifying Failures and Understanding Dependencies 🔬# Simplifying Failures: Making Bugs Easier to Find 🔬

---

## Why Simplify?

- Big, messy bugs are **hard to understand**
- Smaller, simpler failures are **easier to debug**
- Goal: **Isolate the minimal input or code** that still causes the bug

<!--
- Simplifying a failing case helps you focus on the root cause of a bug.
- Smaller, simpler failures are easier to analyze and fix.
- Removing unrelated complexity speeds up debugging.
-->

---

## Delta Debugging: Automated Simplification

- A method to **automatically reduce** failing inputs or code
- Repeatedly removes parts and tests if the bug still happens
- Finds a **minimal failure-inducing input**

### Example Scenario

- A 1000-row CSV causes a crash
- Delta debugging finds that **just 3 rows** are enough to trigger it
- Now, debugging is much easier!

<!--
- Delta debugging is an automated way to reduce failing inputs or code.
- It helps isolate the minimal cause of a bug quickly.
- This technique saves time by focusing only on what triggers the failure.
-->

---

## How to Simplify Failures Manually

- **Reduce input data**: smaller files, fewer rows, simpler cases
- **Comment out code**: remove parts unrelated to the bug
- **Use fixed/random seeds**: make failures reproducible
- **Isolate functions**: test parts separately

### Example

```python
# Original: crashes with full dataset
data = load_data("bigfile.csv")

# Simplify: test with a tiny sample
data = data.head(5)
```

<!--
- Manually simplifying failures often reveals the exact data or code causing the problem.
- Testing smaller inputs or isolated functions helps pinpoint bugs.
- This approach makes debugging more manageable and effective.
-->

---

## Benefits of Smaller Failures

- **Faster debugging**: less to analyze
- **More reproducible**: fewer variables
- **Easier to share**: others can help with a small example
- **Better understanding**: focus on the root cause

<!--
- Creating a minimal, reproducible example makes bugs easier to understand and fix.
- Smaller examples reduce distractions and variables.
- They also help others assist you more effectively.
-->

---

# Next: Understanding Dependencies and Program Flow 🔄# Understanding Dependencies and Program Flow 🔄

---

## Why Care About Dependencies?

- Bugs often hide in **unexpected interactions** between parts of code
- Knowing **which parts affect what** helps you find the root cause
- Like tracing symptoms back to the source in medicine

<!--
- Understanding how parts of your code depend on each other helps locate bugs.
- Bugs often hide in unexpected interactions between components.
- Mapping dependencies can narrow down the source of problems.
-->

---

## Control Flow vs Data Flow

### Control Flow

- The **order** in which code runs
- Controlled by `if`, `for`, `while`, function calls
- Visualized as a **flowchart**

Example:

```python
if patient_age > 65:
    risk = "high"
else:
    risk = "low"
```

<!--
- Control flow is the order in which your code runs, controlled by conditions and loops.
- It determines which parts of your program execute in different situations.
- Visualizing control flow helps understand program behavior and find logic errors.
-->

---

### Data Flow

- How **data moves** through variables and functions
- Tracks **where values come from and go**
- Helps find **where bad data originates**

Example:

```python
bmi = weight / (height ** 2)
category = categorize_bmi(bmi)
```

<!--
- Data flow tracks how data moves and changes throughout your program.
- Understanding data flow helps identify where incorrect values originate.
- Analyzing data flow is key to diagnosing many bugs.
-->

---

## Identifying Relevant Code

- When debugging, focus on **code that influences the bug**
- Ignore unrelated parts to save time
- Use print/logging to trace **which variables affect the failure**

<!--
- Focusing on the code and variables directly related to a bug saves time.
- Ignoring unrelated parts reduces overwhelm during debugging.
- This targeted approach speeds up finding and fixing issues.
-->

---

## Program Slicing 🥒

- Technique to extract **only the code affecting a specific value or line**
- Like cutting out a slice of the program relevant to the bug
- Can be done manually or with tools

Example:

- Bug in `risk_score`
- Slice includes all code that **sets or modifies** `risk_score`

<!--
Slicing helps isolate the minimal code responsible for a bug.
-->

---

# Next: Debugging Workflows and Strategies 🧭# Debugging Workflows and Strategies 🧭

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

# Next: Debugging Examples, Challenges, and Practice 🧩# Debugging Examples, Challenges, and Practice 🧩

---

## Challenge 1: Fixing a String Split Function ✂️

### The Buggy Code

```python
def split_word_in_two(to_split):
    length = len(to_spit)
    half_length = length / 2
    part1 = to_split[:half]
    part2 = to_split[half:]
    return part1, part2
```

### Issues

- Typo: `to_spit` instead of `to_split`
- `half_length` is a float, should be integer
- Uses undefined variable `half`

### Fixes

```python
def split_word_in_two(to_split):
    length = len(to_split)
    half = length // 2
    part1 = to_split[:half]
    part2 = to_split[half:]
    return part1, part2
```

<!--
Common beginner bugs: typos, wrong variable names, integer division.
-->

---

## Challenge 2: Debugging Insertion Sort 🔢

### The Buggy Code (snippet)

```python
for i in range(1, to_sort):
    # ...
    while j > 0 and to_sort[j-1] > to_sort[j]
        # swap logic
```

### Issues

- `range(1, to_sort)` should be `range(1, len(to_sort))`
- Missing colon `:` after `while` condition
- Indentation and swap logic errors

### Fixes

```python
for i in range(1, len(to_sort)):
    j = i
    while j > 0 and to_sort[j-1] > to_sort[j]:
        to_sort[j], to_sort[j-1] = to_sort[j-1], to_sort[j]
        j -= 1
```

<!--
Highlights importance of syntax, loop ranges, and careful swapping.
-->

---

## Live Demo! 🎥

<!--
Live demos make debugging tangible and less intimidating.
-->

---

## Assignment Overview 📝

- **Goal:** Practice debugging real Python scripts
- **Tasks:**
  - Fix provided buggy functions
  - Add print/logging to trace issues
  - Use `pdb` or IDE debugger
  - Simplify inputs to isolate bugs
- **Deliverable:** Submit fixed scripts and a brief reflection

<!--
Assignments reinforce skills and build debugging confidence.
-->

---

# End of Debugging Section — Next: Working with Big Data 📊# Working with Data Bigger Than Memory 📊🐘

---

## Why Does This Matter?

- Health data can be **huge**: millions of patients, years of records
- Your laptop's RAM is **limited**
- Need tools and techniques to **analyze big data efficiently**

<!--
Big data challenges are common in health research, requiring scalable solutions.
-->

---

## Tools for Big Data

| Tool     | Key Features                                   | When to Use                          |
|----------|-----------------------------------------------|-------------------------------------|
| **pandas**  | Easy, powerful, in-memory                   | Small to medium datasets            |
| **polars**  | Fast, Rust-based, lazy evaluation           | Larger datasets, faster processing  |
| **spark**   | Distributed, scales to clusters             | Very large, distributed data        |
| **duckdb**  | SQL database, fast analytics                | SQL queries, medium-large data      |
| **dask**    | Parallel, pandas-like API                   | Larger-than-memory, parallel tasks  |

<!--
Choose tools based on data size, complexity, and available resources.
-->

---

## pandas vs polars 🐼⚡

- **pandas 3.0** moving to **Arrow** backend for better performance
- **polars** built on Arrow from the start, supports **lazy** and **streaming** APIs
- polars often **faster** and uses **less memory**
- Both have **similar syntax**, so switching is easy

Example:

```python
import polars as pl

df = pl.read_csv("bigfile.csv")
result = df.filter(pl.col("age") > 65).groupby("diagnosis").count()
```

<!--
polars enables faster, scalable data processing with familiar syntax.
-->

---

## polars vs pandas Syntax Quick Review 📝

| **Task**                     | **pandas**                                         | **polars**                                         |
|------------------------------|----------------------------------------------------|----------------------------------------------------|
| Import                      | `import pandas as pd`                              | `import polars as pl`                              |
| Read CSV                    | `pd.read_csv("file.csv")`                          | `pl.read_csv("file.csv")`                          |
| Lazy read CSV               | *not supported*                                   | `pl.scan_csv("file.csv")`                          |
| Select columns              | `df[['age', 'sex']]`                              | `df.select(['age', 'sex'])`                        |
| Filter rows                 | `df[df['age'] > 65]`                              | `df.filter(pl.col('age') > 65)`                    |
| Group by and aggregate      | `df.groupby('diagnosis').size()`                  | `df.groupby('diagnosis').count()`                  |
| Chain operations            | `df[df['age'] > 65].groupby('diagnosis').size()`  | `df.filter(pl.col('age') > 65).groupby('diagnosis').count()` |
| Save CSV                    | `df.to_csv("out.csv")`                            | `df.write_csv("out.csv")`                          |

<!--
polars syntax is similar to pandas, but uses `pl.col()` for expressions and supports lazy queries.
-->

---

## When pandas Fails but polars Works 🚫🐼✅⚡

### Scenario

- You have a **20GB CSV** of patient records
- Your laptop has **8GB RAM**
- You want to filter and summarize the data
---

## Why Parquet Format Matters 🗄️

- **Parquet** is a popular **columnar storage format** optimized for big data analytics.
- Stores data **compressed and efficiently**, saving space.
- Allows reading **only needed columns**, speeding up queries.
- Supports **partitioning** datasets by columns (e.g., year, diagnosis) for scalable processing.
- Works well with **polars, pandas, Spark, DuckDB, Dask**.
- Prefer Parquet over CSV for large, frequently accessed datasets.

<!--
- Parquet is a best practice format for big data analytics.
- Columnar storage and compression make it faster and smaller than CSV.
- Partitioning Parquet files by columns (e.g., year, diagnosis) enables scalable, selective queries.
- Widely supported by modern data tools like polars, pandas, Spark, DuckDB, and Dask.
-->
---

### pandas Attempt

```python
import pandas as pd

df = pd.read_csv("patients_20GB.csv")  # tries to load entire file
# Likely causes MemoryError or system freeze
```

- **Problem:** pandas loads the **whole file into memory**
- **Result:** Crashes or becomes unresponsive

### polars Solution

```python
import polars as pl

result = (
    pl.scan_csv("patients_20GB.csv")  # lazy, no full load
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect(streaming=True)  # processes in chunks
)
```

- **Why it works:** polars **streams data in chunks**, never loads all at once
- **Result:** Finishes successfully, even on a laptop

<!--
polars' lazy and streaming execution enables analysis of datasets far larger than RAM, where pandas fails.
-->

---

## Using polars with Larger-than-Memory Data 🧠💾

### Lazy Evaluation

- polars can **build a query plan** without loading all data
- Only loads data **when needed** for the final result
- Use `.scan_csv()` instead of `.read_csv()` for big files

Example:

```python
import polars as pl

lazy_df = pl.scan_csv("bigfile.csv")
result = (
    lazy_df
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect()  # triggers execution
)
```

<!--
Lazy queries let polars optimize and avoid loading unnecessary data.
-->

---

### Streaming Execution

- polars can **process data in chunks** (streaming)
- Avoids loading entire dataset into memory
- Enabled automatically with `.scan_csv()` and `.collect(streaming=True)`

Example:

```python
result = (
    pl.scan_csv("bigfile.csv")
    .filter(pl.col("age") > 65)
    .groupby("diagnosis")
    .agg(pl.count())
    .collect(streaming=True)
)
```

<!--
Streaming allows analysis of datasets much larger than RAM.
-->

---

### Chunked Reading

- For very large files, read in **chunks** and process incrementally
- Combine results after processing each chunk

Example:

```python
for batch in pl.read_csv("bigfile.csv", batch_size=100_000):
    # process each batch separately
    pass
```

<!--
Chunking is useful when lazy/streaming is not enough or for custom workflows.
-->

---

## Larger-than-Memory Operations

### What's Easy?

- **Parallelizable** tasks:
  - Calculating mean, sum, counts
  - Filtering rows independently
- **Scan-based** operations:
  - Reading data in chunks

### What's Hard?

- **Non-partitioned joins**
- **Complex groupbys**
- **Sorting entire datasets**

### Strategies

- **Chunking:** process data in pieces
- **Out-of-core algorithms:** work without loading all data at once
- **Distributed frameworks:** use multiple machines (e.g., Spark)

<!--
Understanding what scales well helps design efficient analyses.
-->

---

## Live Demo!

---

## Assignment: Big Data Health Analysis 🏥📈

### Task
See demo: [`demo4-bigdata`](lectures/02/demo/demo4-bigdata.md)

- Use polars (or dask) to analyze a large health dataset (provided or generated)
- Perform:
  - Filtering (e.g., age, diagnosis)
  - Grouping and aggregation
  - Save results to a file

### Requirements

- Use **lazy evaluation** (`scan_csv`) and/or **streaming**
- Avoid loading entire dataset into memory
- Include code comments explaining steps
- Write a brief reflection:
  - Challenges faced
  - How polars helped
  - What you learned

### Bonus

- Try chunked reading and incremental processing
- Compare with pandas (if feasible)

<!--
Assignment builds practical skills for scalable health data analysis.
-->

---

## Resources

- [pandas documentation](https://pandas.pydata.org/)
- [polars documentation](https://pola.rs/)
- [dask documentation](https://dask.org/)
- [duckdb documentation](https://duckdb.org/)
- [Spark documentation](https://spark.apache.org/)
