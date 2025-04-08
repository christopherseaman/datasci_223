# Lecture 2 Outline: Jupyter, Debugging, and Big Data

---

## 0. Jupyter Notebooks Basics

- What is Jupyter? Interactive coding environment
- **Magic Commands**
  - `%pip` for package management inside notebooks
  - `%timeit`, `%debug`, `%run`, `%load`, `%store`
- **Shell Commands**
  - Using `!` to run shell commands (e.g., `!ls`, `!pwd`)
- **Best Practices**
  - Keeping notebooks clean and reproducible
  - Exporting notebooks to scripts
- **Resources**
  - [Dataquest Jupyter tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

---

## 1. Debugging Python Code

### 1.1 Debugging Mindset

- Programming is doing things that don't work until they do
- Bugs are inevitable; debugging is a core skill
- The scientific method in debugging
- Rubber duck debugging: explain your code aloud
- Minimize and isolate failures

### 1.2 Common Python Errors

- SyntaxError, IndentationError
- NameError, AttributeError
- FileNotFoundError, IndexError
- ImportError, TypeError, ValueError
- Strategies to interpret and fix errors

### 1.3 Observing Program Behavior

- Print statements and logging
- Using assertions
- Collecting failure data
- Reproducing bugs reliably
- The role of test cases

### 1.4 Debugging Tools Overview

- **In-code debugging**
  - Conditional debug code (`if DEBUG`)
  - Print statements
  - `break` and `exit`
  - Logging (levels, configuration)
- **Console debugging**
  - `pdb` basics
  - Setting breakpoints with `pdb.set_trace()` or `breakpoint()`
  - Commands: `n`, `s`, `c`, `q`, `p`, `l`
- **IDE Debugging (VS Code)**
  - Setting breakpoints
  - Stepping through code
  - Variable inspection
  - Watch expressions
  - Debug console
  - Configuring `launch.json`
- **Advanced tools (brief mention)**
  - ipdb, pudb, remote debugging
  - Profiling and performance debugging
  - Linters and static analysis

### 1.5 Simplifying Failures

- Delta debugging principles
- Minimizing failure-inducing inputs
- Isolating causes by input reduction
- Benefits of smaller failure cases

### 1.6 Understanding Dependencies

- Data flow and control flow basics
- Identifying relevant code for a failure
- Program slicing concepts

### 1.7 Debugging Workflows

- Setting and managing breakpoints
- Stepping and inspecting variables
- Using conditional breakpoints
- Call stack exploration
- Using test cases to reproduce bugs

### 1.8 Debugging Examples and Challenges

- **Challenge 1:** Fixing a buggy string splitting function
- **Challenge 2:** Debugging insertion sort implementation
- **Demos:** Using VS Code debugger: setting breakpoints, stepping, inspecting variables, call stack, conditional breakpoints
- **Live Demo:** Debugging a small script in Jupyter and/or VS Code
- **Assignment:** Use VS Code debugging tools (breakpoints, stepping, variable inspection, call stack) to fix provided buggy scripts
- **Assignment:** Debugging exercises with provided buggy scripts

### 1.9 References

- freeCodeCamp Debugging Handbook
- The Debugging Book (selected chapters)
- VS Code Python debugging docs
- Guide to Debugging Python code
- Links to tutorials and videos

---

## 2. Working with Data Bigger Than Memory

### 2.1 Motivation

- Datasets often exceed RAM capacity
- Need for scalable data processing

### 2.2 Tools Overview

- **pandas**: traditional in-memory data analysis
- **polars**: fast, Rust-based, lazy evaluation
- **spark**: distributed big data processing
- **duckdb**: embedded analytics database
- **dask**: parallel computing with pandas-like API

### 2.3 pandas vs polars

- pandas 3.0 moving to Arrow backend
- polars built on Arrow, supports lazy & streaming APIs
- Performance and scalability considerations

### 2.4 Larger-than-memory operations

- What’s easy:
  - Parallelizable, scan-based computations (mean, mode)
- What’s hard:
  - Non-partitioned joins and merges
  - Complex groupbys and sorts
- Strategies:
  - Chunking data
  - Out-of-core algorithms
  - Using distributed frameworks

### 2.5 Demo and Assignment

- Example: Aggregating a large CSV with polars or dask
- Assignment: Process a dataset larger than memory using one of the tools

### 2.6 References

- Official docs for pandas, polars, spark, duckdb, dask
- Tutorials on scalable data processing

---

## Structure of Lecture Materials

- `lectures/02/parts/`
  - `jupyter.md` — Jupyter basics notes
  - `debugging/`
    - `overview.md` — Debugging mindset, errors, observing
    - `tools.md` — Debugging tools (print, logging, pdb, IDE)
    - `simplifying.md` — Minimizing failures, delta debugging
    - `dependencies.md` — Data/control flow, slicing
    - `workflows.md` — Debugging workflows
    - `examples.md` — Examples, challenges, demos
    - `demos/` — Demo scripts (`.py`) and instructions (`.md`)
    - `assignments/` — Assignment descriptions and starter code
  - `bigdata/`
    - `notes.md` — Working with big data notes
    - `demos/` — Demo scripts and instructions
    - `assignments/` — Assignment descriptions and starter code

---

This outline will guide the creation of detailed lecture notes, demos, and assignments.