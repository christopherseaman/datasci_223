# Debugging Examples, Challenges, and Practice 🧩

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

## Live Demo Ideas 🎥

- Debugging a small health data script
- Use print/logging to trace variables
- Set breakpoints in VS Code
- Step through code, fix a bug live
- Show how simplifying inputs helps

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

# End of Debugging Section — Next: Working with Big Data 📊