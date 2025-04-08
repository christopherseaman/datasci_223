# Jupyter Notebooks: Your Interactive Coding Playground 🧑‍💻📓

---

## What is Jupyter?

Jupyter notebooks are interactive documents that let you **write and run code, see results immediately, and mix in text, images, and equations**.

- Great for **exploring data**, **trying out code**, and **sharing your work**.
- Widely used in **health data science** for analysis, visualization, and reporting.

<!--
Instructor note:
- Emphasize that Jupyter is beginner-friendly and forgiving.
- Encourage students to experiment without fear.
- Mention that notebooks can be exported as reports or scripts.
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
Instructor note:
- Demo `%pip` to install a package.
- Show `%timeit` on a simple calculation.
- Mention `%debug` will be covered more during debugging.
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
Instructor note:
- Explain this is like typing commands in the terminal.
- Useful for checking files, running scripts, managing data.
-->

---

## Best Practices for Notebooks 🧹

- **Keep it clean:** Remove failed code, unnecessary outputs.
- **Use Markdown cells** for explanations, titles, and notes.
- **Restart and run all** before sharing to ensure reproducibility.
- **Export notebooks** as HTML or PDF for reports.
- **Convert to scripts** (`File > Export`) for production code.

<!--
Instructor note:
- Encourage students to narrate their analysis with Markdown.
- Explain reproducibility: others (or future you) can follow the steps.
-->

---

## Resources 📚

- [Dataquest Jupyter tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
- [Jupyter documentation](https://jupyter.org/documentation)
- [Cheat sheet](https://www.datacamp.com/blog/jupyter-notebook-cheat-sheet)

---

## Demo Break #1: Getting Comfortable with Jupyter (10 minutes)

**Goal:** Run some code, use magics, and shell commands.

### Steps:

1. Open a new Jupyter notebook.
2. Run `print("Hello health data science!")`
3. Use `%pip` to install `pandas`.
4. Use `!ls` to list files.
5. Try `%timeit 2 + 2`
6. Add a Markdown cell with your name and today's date.

**Expected Outcome:** Students can run code, use magics, and shell commands confidently.

<!--
Instructor note:
- Walk through steps live.
- Encourage students to experiment.
- Help troubleshoot any issues.
-->

---

# Next: Debugging Python Code 🐛🔍