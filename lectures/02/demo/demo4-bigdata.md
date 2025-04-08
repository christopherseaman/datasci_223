# Demo: Analyzing Large Health Data with polars and pandas 🏥📊

---

## Goal

Compare polars and pandas when analyzing a large health dataset.

---

## Setup

- Use `generate_large_health_data.py` to create `patients_large.csv`
- Use `analyze_large_health_data.py` with `--backend polars` and `--backend pandas`

---

## Tasks

1. **Generate data:**

```bash
python generate_large_health_data.py
```

2. **Analyze with polars:**

```bash
python analyze_large_health_data.py --backend polars
```

3. **Analyze with pandas:**

```bash
python analyze_large_health_data.py --backend pandas
```

4. **Observe:**
   - polars should succeed quickly
   - pandas may crash or be very slow

5. **Inspect output `summary.csv`**

---

## Expected Outcomes

- Students see how polars handles big data efficiently
- Students understand pandas' memory limitations
- Students learn to choose the right tool for big data

---

## Notes

<!--
This demo highlights the practical benefits of polars' lazy and streaming execution for large datasets.
-->