# Simplifying Failures: Making Bugs Easier to Find 🔬

---

## Why Simplify?

- Big, messy bugs are **hard to understand**
- Smaller, simpler failures are **easier to debug**
- Goal: **Isolate the minimal input or code** that still causes the bug

<!--
Simplifying failures helps focus on the root cause, not distractions.
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
Delta debugging saves time by shrinking the problem to its core.
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
Simplification often reveals the specific data or code causing the issue.
-->

---

## Benefits of Smaller Failures

- **Faster debugging**: less to analyze
- **More reproducible**: fewer variables
- **Easier to share**: others can help with a small example
- **Better understanding**: focus on the root cause

<!--
Always try to create a minimal, reproducible example when debugging.
-->

---

# Next: Understanding Dependencies and Program Flow 🔄