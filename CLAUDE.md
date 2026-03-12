# AGENTS

Applied Data Science with Python course materials (UCSF DataSci 223, Spring 2025). Documentation-heavy repo for beginners; lectures are long-form Markdown with demos. Adapt/update previous year's content from `refs/lectures_25/` when possible.

**Course structure**: Applied survey course—each week covers a single topic, either standalone (e.g., SQL) or as part of a series building complexity (classification → neural networks → LLMs → LLM API/agentic/workflows).

**GitHub Pages site URL**: `https://christopherseaman.github.io/datasci_223/` (deployed via Eleventy; images use relative paths from the referring document, e.g., `media/xkcd_selection_bias.png` from `03/lecture_03.md`)

**IDE**: VS Code is the common IDE for this class.

**Audience**: Health data science masters students who are beginners in programming (Python, git, shell commands).

## Source of truth

- `CLAUDE.md` (aliased as `AGENTS.md`) is the source of truth for agent work in this repo.

## Repository layout

- `XX/`: Current year's lecture directories (01–10), each containing `lecture_XX.md`, `media/`, `demo/`, `assignment/`.
- `_includes/`: Eleventy templates and inline CSS (`layout.njk`, `sidebar.njk`).
- `_data/nav.js`: Site navigation structure (lectures, links, tracks).
- `.eleventy.js`: Eleventy build config (passthrough copy, computed titles/permalinks, collections).
- `package.json`: Node dependencies (Eleventy, syntax highlighting, markdown-it plugins).
- `refs/`: Reference material — prior year lectures (`refs/lectures_24/`, `refs/lectures_25/`), data links, PhysioNet notes, SQL guide, debugging tips, etc.
- `scripts/`: Utilities (e.g., `fetch_xkcd_2x.py` for downloading XKCD comics).
- `refs/all_xkcd.csv`: Index of available XKCD comics for reference.

## Build and preview

- **Site (Eleventy)**: Node.js 24+. `npm install`, then `npm start` for local preview or `npm run build` for static output to `_site/`.
- **Demos/assignments (Python)**: Python 3.12. **Prefer `uv`** for local dev (`uv venv .venv && uv pip install -r requirements.txt`); student default is `python -m venv`.
- Deploy: CI auto-deploys on push to main via `.github/workflows/deploy.yml` (Eleventy build + GitHub Pages artifact).

## Authoring workflow

- **Heading hierarchy**: Each lecture has a single `# Title` (rendered as `<h1>`), with content sections using `##` and subsections using `###`/`####`.

### Lecture style (student-facing)

- **Primary goal**: students should be able to read/scan the lecture and understand concepts on first exposure.
- **Beginner-friendly**: avoid jargon; define terms when introduced. Build knowledge incrementally.
- **Visual learning**: prefer diagrams, screenshots, and concrete outputs over walls of text.
- **Content balance**: each major topic should blend conceptual foundations, reference material, practical examples, and hands-on demos.
- **Tone**: professional and clear; informal/fun is fine, but not a script. The lecturer speaks *to* the content; the content itself should not read like speaking notes.
- **Humor & emojis**: allowed and encouraged when used sparingly and placed between relevant sections/sub-sections; avoid making the *core* explanation itself jokey in a way that obscures meaning.
- **No time estimates**: do not include time cues/estimates for sections, demos, or assignments.

### Lecture formatting

- Title is a single `# Title` heading; main sections use `##`, subsections `###`/`####` (never bolded fake headings).
- Prefer concise bullets over long prose with 4-space indents for nesting.
- Each major section should usually include:
  - freeform intro — can be multiple paragraphs; this is where concepts are explained, analogies drawn, and connections to previous lectures made. Prior years' lecture content (see `refs/lectures_25/` and `refs/lectures_24/`) is a good source of content to adapt here.
  - a visual/table/output (or `#FIXME` placeholder)
  - a `### Reference Card: ...` table
  - a short `### Code Snippet: ...`
- Place visuals before code.
- Mark demo breaks only with `## LIVE DEMO!`, `## LIVE DEMO!!`, `## LIVE DEMO!!!` (first/second/third demo). Each marker should correspond to a concrete walkthrough in `XX/demo/`.
- Skip "Summary" sections.

### Reference card formats

Two standard formats for reference cards:

**Multi-method reference table** — for related methods that don't need dedicated subsections:

```markdown
### Reference Card: spaCy Basics

| Category | Method / Attribute | Purpose & Arguments | Typical Output |
| :--- | :--- | :--- | :--- |
| **Setup** | `spacy.load("model_name")` | Loads a language pipeline (e.g., `"en_core_web_sm"`). | Language object (`nlp`) |
| **Process** | `nlp("Text string")` | Processes text through the pipeline. | `Doc` object |
| **Tokenization** | `[token.text for token in doc]` | Breaks `Doc` into individual words/punctuation. | `List[str]` |
| **Linguistic** | `token.pos_` | Returns the Part-of-Speech tag (e.g., NOUN, VERB). | `String` |
| **Entities** | `doc.ents` | Accesses named entities found in the text. | `Span` objects |
```

**Single-function deep-dive** — for complex functions needing detailed documentation:

```markdown
### Reference Card: `spacy.load()`

| Component | Details |
| :--- | :--- |
| **Signature** | `spacy.load(name, *, vocab=True, disable=[], exclude=[], config={})` |
| **Purpose** | Loads a trained pipeline package by name and returns a `Language` object. |
| **Parameters** | • **name** (str): The package name or path to data directory.<br>• **disable** (list): Names of pipeline components to ignore (e.g., `['tagger', 'parser']`).<br>• **exclude** (list): Pipeline components to skip entirely and never load.<br>• **config** (dict): Overrides for model configuration. |
| **Returns** | `Language`: A stateful container that processes text and returns `Doc` objects. |
```

### Assets and links

- Assets live alongside their lecture folder (e.g., `01/media/...`).
- Images use relative paths from the referring document (e.g., `media/xkcd_selection_bias.png` from `03/lecture_03.md`), NOT absolute paths.
- Prefer local comics/images. If reusing from elsewhere, copy into the lecture's `media/` subdir first.
- XKCD helper: `scripts/fetch_xkcd_2x.py` downloads comics via explainxkcd file pages (2x "Original file" links). Usage: `./scripts/fetch_xkcd_2x.py 1597:Git 1722:Debugging:xkcd_debugging.png`

### Notes

- Place speaking notes in `NOTES.md` with matching headings after the lecture text is complete.
- Notes should contain the actual things to say — short talking points, anecdotes, clarifications, common mistakes to call out — not instructions like "talk about X" or restatements of what's already in the lecture.
- No inline HTML comment speaking notes in the lecture text.

## Demo structure and conventions

Lectures include 3 hands-on demos at ~1/3, ~2/3, and end of 90-minute session. Demos build on lecture content with realistic complexity.

**Content philosophy:**

- **Lecture code blocks:** Short, simple, minimal examples demonstrating single concepts
- **Demo code:** Realistic complexity with health data, multiple steps, edge cases—mirrors real-world usage
- **Progressive difficulty:** Start simple, build complexity across the three demos
- Demos should be completable in a short session with clear success checkpoints
- Include ways to confirm completion (visual/tabular output after each step)

**File naming convention:** `0Xy_description.suffix`

- `X` = demo number (1, 2, 3)
- `y` = order within demo (a, b, c, ...; omit if single file)
- `description` = very short descriptor (1-3 words)
- `suffix` = file type (`.md`, `.py`, `.ipynb`, `.yaml`, etc.)

**Markdown → Jupyter conversion:**

- Write demos as `.md` files (easier to review, git-friendly)
- Convert with `jupytext --to notebook demo/*.md` before class
- Use jupytext percent format (`#%%`) or markdown format for cells

## Assignment structure

Weekly assignments are **pass/fail** and should be straightforward for students who understand the lecture content. Coursework uses GitHub Classroom with pytest-based autograding via GitHub Actions.

```
XX/assignment/
├── README.md                 # Instructions
├── {source_files}.py         # Starter/solution code
├── hints.md                  # Optional hints
└── .github/
    ├── tests/test_*.py       # pytest test suite
    └── workflows/classroom.yml
```

### Assignment testing philosophy

**Test behaviors and artifacts, not implementation details.** Students may solve problems in many valid ways (Gödel incompleteness applies to grading too).

**Good tests check:**

1. **Code execution**: Does the code run without errors?
2. **Artifacts generated**: Are output files created with correct format/content?
3. **Function behavior**: Do imported functions produce correct results with known inputs?
4. **Known input/output pairs**: Test with fixtures, verify expected outputs

**Bad tests check:**

- Specific code patterns ("import logging" string matching)
- Function names or variable names (students may name differently)
- Code structure or style (unless that's the learning objective)

**For notebooks:**

- Execute with `nbconvert --execute` and check exit code
- Read outputs from executed notebook cells
- Or import functions from converted `.py` and test directly

## Validation protocol (agent checklist)

Do not declare work complete without validation.

### Lecture ↔ demo alignment

- `XX/lecture_XX.md` includes exactly three break markers: `## LIVE DEMO!`, `## LIVE DEMO!!`, `## LIVE DEMO!!!`.
- Each break marker has a corresponding, concrete walkthrough in `XX/demo/`.
- Lecture does not embed full demo walkthrough steps (those live in `XX/demo/`).

### Demo correctness (including intentional failures)

- Demos run end-to-end without errors unless an error is explicitly introduced for pedagogy.
- For intentionally buggy demos:
  - document the expected failure mode clearly in the demo guide/walkthrough (what error, where it happens, what students should learn)
  - keep the "buggy" artifact and the "fixed" artifact side-by-side (e.g., `02a_*.md` and `02b_*.md`).

**Minimum validation for a demo change:**

- If a demo includes a `.py` script: run it.
- If a demo includes a `.md` notebook source: convert (if needed) and execute the `.ipynb` via `jupyter nbconvert --execute`.

### Assignment alignment and completability

- `XX/assignment/README.md` is solvable using only lecture content + demos.
- Assignment prompts match autograder expectations (required artifacts, behaviors, fixed strings if any).

### Scratch-dir completion test (recommended when assignments change)

Test what a student does from a clean copy of the assignment folder.

- Copy `XX/assignment/` to a fresh temp directory.
- Run the student workflow there (create venv, install deps, run scripts/notebooks, generate artifacts).
- Run the autograder tests from inside the copied assignment directory.

**Key property:** tests should pass without relying on repo-global context outside the assignment folder.

**Practical pattern (example):**

- Create scratch dir: `tmp_dir=$(mktemp -d)`
- Copy: `cp -R XX/assignment "$tmp_dir"`
- Run tests from scratch: `python -m pytest .github/tests -q`

### When to run the Eleventy build

- Do not run `npx @11ty/eleventy` as routine validation.
- Run it only when changing Eleventy configuration, templates, or site navigation (`.eleventy.js`, `_includes/`, `_data/nav.js`).

## Engineering guardrails (per ~/.claude/CLAUDE.md)

- Configuration-first: do not hardcode values that belong in config; prefer centralized settings and utility modules over duplication.
- DRY and separation of concerns: extract shared logic into helpers; keep orchestration separate from business logic and from data/visual layers.
- Simplicity and clear data flow: favor explicit, minimal abstractions with named inputs/outputs; avoid hidden side effects.
- Naming and comments: descriptive function/variable names; brief contextual comments only when logic is non-obvious.
- **Testing/validation required**: never declare work done without actually running the code AND validating output. Run the lecture/demo/assignment validation appropriate to what changed (see "Validation protocol (agent checklist)" above). If testing is impossible, document assumptions and provide user-run verification steps.
- Environment discipline: prefer `uv` for speed, fallback to venv/conda; stable working dir, relative paths; avoid complex one-liners—script it instead.
- Git hygiene: direct, professional commit messages—no cute co-author tags (e.g., "Co-Authored-By: Claude"), no emoji, no "Generated with" footers. Conventional commits preferred.
