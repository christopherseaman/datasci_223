# Debugging References Inventory

This document provides a per-file outline of the contents in `refs/debugging/` to support lecture development.

---

## challenge.md
- **Debug Challenge 1:** Debug a buggy function `split_word_in_two`
- **Hints:**
  - Use IDEs like Spyder or VS Code
  - Read interpreter output carefully
  - Use print statements and variable viewers
  - Multiple bugs may exist; fix incrementally
- **Buggy Code:**
  - Function to split a word into two halves
  - Contains typos (`to_spit`), wrong variable names (`half` vs `half_length`), integer division issues
- **Includes:** Link to a YouTube walkthrough

---

## challenge2.md
- **Debug Challenge 2:** Fix a broken insertion sort implementation
- **Context:**
  - Explains insertion sort conceptually
- **Buggy Code:**
  - Syntax errors (missing colon, wrong range usage)
  - Logic errors in loop conditions and swapping
- **Goal:** Practice debugging nested loops and sorting logic

---

## debugging_handbook.md
- **Author:** Samyak Jain (freeCodeCamp)
- **Sections:**
  - **Common Python Errors:** SyntaxError, IndentationError, NameError, AttributeError, FileNotFoundError, IndexError, ImportError, TypeError, ValueError
  - **Foundational Debugging Techniques:**
    - Print statements
    - Logging (levels, configuration)
    - Exception handling (`try`/`except`)
    - Assertions
  - **Advanced Debugging:**
    - Unit testing (`unittest`, `pytest`)
    - Interactive Debugger (`pdb`)
    - Remote debugging
    - Profiling (cProfile, snakeviz, line_profiler, memory_profiler)
    - Linters and analyzers (pylint, flake8, black, mypy)
  - **IDE Debugging Features:**
    - Breakpoints, stepping, call stack, variable inspection
    - Conditional breakpoints, watch expressions
    - Debugging in Jupyter notebooks
  - **Performance Debugging**
  - **Debugging Mindset:**
    - Rubber duck debugging
    - Version control and git bisect
    - Effective search strategies
    - Using forums and documentation
- **Includes:** Code examples, screenshots, links

---

## guide.md
- **Title:** A Guide to Debugging Python code (and why you should learn it)
- **Topics:**
  - Why debugging is essential
  - Types of debugging tools:
    - Standalone UI apps (PuDB, Winpdb)
    - Code-in breakpoints (`breakpoint()`)
    - IDE debuggers
  - **Example Debugging Session:**
    - Pydantic model with API response
    - Using `breakpoint()` and `pdb` to inspect data
    - Fixing model field mismatches
  - **Debugging in IDEs (PyCharm focus):**
    - Setting breakpoints
    - Variable inspection
    - Stepping through code
    - Evaluate expressions tool
  - **Conclusion:** Embrace debugging as a skill

---

## ms_debugging.md
- **Source:** Microsoft Docs (VS Code)
- **Focus:** Debugging Python in Visual Studio Code
- **Topics:**
  - Python Debugger extension (`debugpy`)
  - Creating and customizing `launch.json`
  - Debugging configurations:
    - Launch vs attach
    - Flask, Django, Pyramid, Gevent
    - Remote debugging via SSH, Docker
  - Command-line debugging with `debugpy`
  - Attaching to running processes
  - SSH tunneling for remote debugging
  - Debugging features:
    - Breakpoints, conditional breakpoints
    - Variable inspection, watch, call stack
    - Auto-reload, subprocess debugging
  - Troubleshooting common issues
  - Links to further resources

---

## testing.md
- **Source:** Microsoft Docs (VS Code)
- **Focus:** Testing Python code in Visual Studio Code
- **Topics:**
  - Configuring test frameworks (`unittest`, `pytest`)
  - Test discovery and patterns
  - Running tests:
    - From editor gutter
    - Test Explorer
    - Command Palette
  - Debugging tests:
    - Breakpoints in tests
    - Debugging individual or all tests
    - Custom debug configurations for tests
  - Test coverage:
    - Using `pytest-cov` or `coverage.py`
    - Viewing coverage in editor and Test Explorer
  - Parallel test execution with `pytest-xdist`
  - Django test integration
  - Test-related commands and settings

---

## debuggingbook-html/ (The Debugging Book)

A multi-chapter interactive book on debugging techniques, exported as HTML files:

### Main Chapters with Subtopics:

- **01_Intro.html — Introduction**
  - What is debugging?
  - Types of bugs
  - The scientific method in debugging
  - Debugging workflow overview
  - Importance of minimizing and isolating failures

- **02_Observing.html — Observing Program Behavior**
  - How to observe failures effectively
  - Adding print statements and logging
  - Using assertions
  - Collecting failure data
  - Reproducing bugs reliably
  - The role of test cases

- **03_Dependencies.html — Dependencies and Data Flow**
  - Understanding program dependencies
  - Data flow analysis basics
  - Control flow vs data flow
  - Identifying relevant code for a failure
  - Program slicing concepts
  - Tools for dependency analysis

- **04_Reducing.html — Simplifying Failure-Inducing Inputs**
  - Delta debugging principles
  - Automated test case minimization
  - Isolating failure causes by input reduction
  - Examples of delta debugging on code and data
  - Benefits of smaller failure cases

- **05_Abstracting.html — Abstracting Failures**
  - Generalizing failure circumstances
  - Finding common patterns in failures
  - Grouping related failures
  - Abstracting test cases for better understanding

- **06_Repairing.html — Automated Program Repair**
  - Techniques for automated bug fixing
  - Generating patches
  - Validating repairs
  - Limitations and challenges

- **07_In_the_Large.html — Debugging Large Systems**
  - Challenges unique to large codebases
  - Managing complexity
  - Modular debugging
  - Tool support for large-scale debugging

- **99_Appendices.html**
  - Supplementary material
  - References and further reading

---

### Additional Topic Pages (selected highlights):

- **Assertions.html** — Using assertions to catch bugs early
- **DeltaDebugger.html** — Details on delta debugging algorithms
- **DynamicInvariants.html** — Inferring likely invariants from executions
- **PerformanceDebugger.html** — Profiling and performance analysis
- **Repairing_Code.html** — Examples of automated repair
- **Slicer.html** — Program slicing techniques
- **StackInspector.html** — Inspecting call stacks
- **StatisticalDebugger.html** — Statistical methods for bug localization
- **Time_Travel_Debugger.html** — Techniques for reversible debugging
- **Tracer.html** — Tracing program execution
- **Tracking.html** — Tracking program state over time

---

### Navigation:
- `00_Index.html`, `00_Table_of_Contents.html`, `index.html` provide book navigation and overview.

---

This updated inventory now includes key subtopics for each Debugging Book chapter, emphasizing foundational concepts most relevant to an introductory lecture.
