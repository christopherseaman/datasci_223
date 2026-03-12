# Interactive demos

- [ ] Adapt demos to open in Google Colab (add "Open in Colab" links to demo pages or lecture nav)
    - Include output from previous execution so students can read without running
    - Remove/stub external dependencies (local files, API keys, etc.) so notebooks are self-contained
    - Demos that are scripts (not notebooks) could use a Pyodide-based embed (e.g., PyScript or a JS-fiddle-style Python runner) instead of Colab
- [ ] Investigate Marimo WASM for lightweight demos that don't need torch/polars/duckdb (Marimo uses Pyodide under the hood)

---

# XKCD duplicate images across earlier lectures

The following xkcd comics are duplicated across earlier lectures (same file, same name, same content hash). Low priority — fix when revising those lectures.

| Comic | Locations |
|---|---|
| `xkcd_machine_learning.png` (`50a207...`) | `02/media/`, `04/media/` |
| `xkcd_classification.png` (`bb7cbf...`) | `01/media/`, `05/media/` |

Options: remove the copy from whichever lecture doesn't reference it, or swap one for a different comic.

---

# Dataset Options for Future Lectures

**Lectures 6-8 (Neural Networks / Deep Learning):**

- `cifar10` - 60k 32x32 color images, 10 classes (keras)
- `fetch_20newsgroups` - ~18k text documents, 20 categories (sklearn)

**MedMNIST options (unused - dermamnist/pneumoniamnist in Lecture 05 demos):**

- `pathmnist` - pathology (colon cancer)
- `chestmnist` - chest X-ray (14 diseases)
- `octmnist` - retinal OCT
- `retinamnist` - fundus camera (diabetic retinopathy)
- `breastmnist` - breast ultrasound
- `bloodmnist` - blood cell microscopy
- `tissuemnist` - kidney cortex microscopy
- `organamnist/organcmnist/organsmnist` - abdominal CT (axial/coronal/sagittal)

# Lecture 02 Follow-up Tasks

- [x] Build lecture visuals (#FIXME graphics: memory vs dataset chart, row/column diagram, updated Polars benchmark, lazy-plan diagram, monitoring screenshot).
- [x] Select and fetch new XKCDs (Data Pipeline/Workflow/etc.) via `scripts/fetch_xkcd_2x.py` and embed them in the lecture.
- [x] Verify lecture media paths (`02/media/...`).
- [ ] Ensure demo artifacts/data generators exist and align with the written instructions (e.g., big CSV generator, dimension tables, pipeline configs).
- [ ] Ensure assignment sample data plus README instructions match the shipped fixtures.
- [ ] Convert each `02/demo/*.md` via Jupytext, execute the notebooks end-to-end, and capture key outputs.
- [ ] Check in the required Jupytext partners (`.ipynb` or percent-format `.py`) after execution so the demos stay synced.
- [ ] Validate the assignment from a scratch directory using the existing `.venv` (`uv run pytest .github/tests -q`).
- [ ] Document demo and assignment validation results for future instructors (logs or summary notes).
- [x] Prep git staging for the updated `02` tree once validations pass (commit/push requested).
- [ ] (Optional later) Add `02/NOTES.md` once lecture content is fully locked.
- [ ] (Optional later) Update `planned_lectures.md` if Lecture 02 scope diverges from the plan.

---

Add sections to NN lecture:

Two questions that I would have liked to address further from lecture:

# **What does adding layers (depth) vs neurons (breadth) do for a neural network?**

Depth creates compositional features while width adds more detectors/features at a single level.

- **Width** (more neurons per layer): universal approximation theorem says one wide hidden layer *can* approximate any function, but may need exponentially many neurons. Easier to train (less vanishing gradient), but less parameter-efficient.
- **Depth** (more layers): learns hierarchical/compositional representations - each layer builds abstractions on the previous. More parameter-efficient for complex functions, but harder to train (vanishing gradients, need skip connections/BatchNorm/etc.).

Interactive resources:

- [**TensorFlow Playground**](https://playground.tensorflow.org/) - Add/remove hidden layers and neurons in real-time, watch decision boundaries evolve, and see how depth vs. width affects learning on toy 2D classification problems.
- [**ConvNetJS**](https://cs.stanford.edu/people/karpathy/convnetjs/) - Karpathy's older but still useful demos. Less polished UI but good for showing training dynamics.

# **LSTM: How does it work and why tanh instead of sigmoid or RelU?**

Short version: my diagram + explanation was not correct:

**"Standard" LSTM = 3 sigmoids (gates) + 1 tanh (candidate) + 1 tanh (on output c_t)**

An LSTM cell has three gates — **forget** (what to discard from cell state), **input** (what new info to store), and **output**(what to emit) — all controlled by sigmoids (0-1 = how much to pass). The cell state acts as a conveyor belt that carries information across time steps with minimal degradation, solving vanilla RNN's vanishing gradient problem.

**Why tanh specifically:**

- **Gates use sigmoid** (0-1 range) - they're *multipliers*, so you want a smooth switch between "block" and "pass"
- **Cell state update and output use tanh** (-1 to +1 range):
    - **Zero-centered outputs** prevent systematic bias in gradient updates (sigmoid outputs are always positive → gradients always same sign → inefficient zigzag optimization)
    - **Negative values matter** for representing "decrease" vs. "increase" in the cell state
    - **ReLU would be problematic** here: unbounded activations risk exploding cell states over many time steps, and dead neurons (negative inputs → zero forever) would permanently erase cell memory

**TL;DR -** sigmoid = gatekeeping (how much), tanh = content (what value, including negative).

Good resources:

- [**Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) -** Excellent diagrams showing data flow through gates
- [**Visualizing memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/) -** Interactive exploration of what RNN/LSTM cells actually memorize
- [**The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) -** Karpathy blog post (it is always Karpathy) showing what LSTM's learn in practice
