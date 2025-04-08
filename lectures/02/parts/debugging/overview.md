# Debugging Python: Mindset, Common Errors, and Observing Behavior 🐛🔍

---

## Why Debugging Matters

Debugging is **figuring out why your code doesn't work** — a core skill for every programmer.

- **Programming = solving problems + fixing mistakes**
- Bugs are **normal**, not a sign of failure
- Debugging is like **detective work**: gather clues, test theories, find the culprit

<!--
Instructor note:
- Normalize bugs as part of learning.
- Emphasize patience and curiosity.
- Use analogy: debugging = medical diagnosis (symptoms, tests, treatment).
-->

---

## The Debugging Mindset 🧠

- **Be systematic:** Change one thing at a time, observe effects
- **Minimize the problem:** Simplify inputs, isolate code
- **Use the scientific method:** Form hypotheses, test, revise
- **Explain it aloud:** Rubber duck debugging 🦆
- **Don't guess wildly:** It wastes time and causes confusion

<!--
Instructor note:
- Encourage students to narrate their thought process.
- Suggest talking to a classmate, or even an actual rubber duck.
- Mention that even experts debug daily.
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
Instructor note:
- Emphasize reading the full error message.
- Show how traceback points to the error line.
- Encourage students to Google error messages.
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
Instructor note:
- Demo adding print statements to buggy code.
- Emphasize reproducibility: "If you can't reproduce, you can't fix."
- Mention that simplifying failures is a key debugging skill.
-->

---

## Demo Break #2: Finding and Understanding Errors (10 minutes)

**Goal:** Practice reading errors and using print statements.

### Steps:

1. Run a provided buggy script (e.g., typo, bad index).
2. Read the error message carefully.
3. Add print statements to see variable values.
4. Fix the bug.
5. Try breaking it again on purpose!

**Expected Outcome:** Students gain confidence reading errors and observing program behavior.

<!--
Instructor note:
- Walk through a simple bug live.
- Encourage students to experiment and break things.
- Help students interpret tracebacks.
-->

---

# Next: Debugging Tools and Techniques 🛠️