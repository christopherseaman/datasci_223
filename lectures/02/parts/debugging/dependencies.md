# Understanding Dependencies and Program Flow 🔄

---

## Why Care About Dependencies?

- Bugs often hide in **unexpected interactions** between parts of code
- Knowing **which parts affect what** helps you find the root cause
- Like tracing symptoms back to the source in medicine

<!--
Understanding dependencies helps narrow down where bugs originate.
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
Control flow determines which parts of code execute based on conditions.
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
Data flow shows how inputs affect outputs, revealing sources of errors.
-->

---

## Identifying Relevant Code

- When debugging, focus on **code that influences the bug**
- Ignore unrelated parts to save time
- Use print/logging to trace **which variables affect the failure**

<!--
Narrowing focus reduces overwhelm and speeds up debugging.
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

# Next: Debugging Workflows and Strategies 🧭