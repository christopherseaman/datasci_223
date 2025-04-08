---
lecture_number: 01
pdf: true
---
<!---
create the lecture notes (projected/shared instead of slides). Let's make the lecture notes in parts, in the lectures/02/parts folder have 1) a .md for the Jupyter part, 2) folder for debugging with a .md for each major topic plus .py scripts and .md's for demos/assignment, 3) folder for working with data bigger than memory with a .md for the notes and .py scripts and .md's for demos/assignment
--->
<!---
- 0) Jupyter notebooks
  - Magic commands, especially %pip
  - ! shell commands
  - https://www.dataquest.io/blog/jupyter-notebook-tutorial/
- 1) Debugging
  - List refs
  - Programming is doing things that don't work over and over until it does
  - Rubberducking
  - debugging tools
    - in-code debugging
      - if DEBUG
      - console.log
      - `break` and `exit`
      - logging
    - console debugging with `pdb` (mention)
      - breakpoints
      - pdb.set_trace()
    - in-IDE debugging with VS Code
  - common issues
    - common error messages
    - malformed/missing data
    - counting errors
  - Examples (minimal dependencies/background knowledge, run in jupyter if possible)
    - Live Demo!
    - Assignment 
- 2) Working with data bigger than memory
  - List refs once topics chosen
  - pandas, polars, spark, duckdb, dask #FIXME: Which best to focus on for students new to this?
  - pandas vs polars?
    - pandas 3.0 → arrow backend
    - polars based on rust, lazy & streaming api's
  - Larger-than-memory operations
    - What's easy? "Ridiculously parallelizable", scan-based (mean, mode)
    - Hard? Non-partitioned joins/merges
--->