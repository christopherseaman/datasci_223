# Lecture 04 Demo Guide

Brief walkthrough for all three NLP demos using Project Gutenberg texts. Each demo corresponds to lecture sections since the previous demo.

**Data:** Excerpts in `data/` (Alice, Holmes, Pride and Prejudice, Frankenstein). See plan for word counts and sizing rationale. Paths, filenames, spaCy model, and chart options are in **`config.yaml`**—change data dir or add texts there without editing notebook code.

**Prep:** Convert markdown to notebooks before class: `jupytext --to notebook 0*.md`

---

## Demo 1: Text Preprocessing Safari

**File:** `01a_preprocessing.md` → `01a_preprocessing.ipynb`
**Lecture sections:** Tokenization, Normalization, Stemming and Lemmatization
**Timing:** After Text Processing Fundamentals (first demo break)
**Text:** Alice in Wonderland, Chapter 1 (~2,000 words)

### Goal
- See how preprocessing choices affect downstream analysis
- Map literary edge cases to clinical-text parallels (abbreviations, negation, non-standard terms)
- Build reusable `preprocess(text)` function

### Walkthrough

1. **Load Alice excerpt:** Read `data/alice_ch1.txt`, show raw text
2. **Tokenize:** Compare `text.split()` vs `nltk.word_tokenize()` vs spaCy on the same sentence. Highlight: "didn't", "rabbit-hole", punctuation
3. **Normalize:** Lowercase (note: loses "Alice" as name signal). Stopwords: experiment keeping vs removing "not"/"no" in "not mad". Connect to "No chest pain" in clinical notes
4. **Stem vs Lemma:** Run on "curiouser", "running", "studies". Show stem can produce non-words; lemma needs POS for "better" → "good"
5. **Build function:** Implement `preprocess(text)` that tokenizes, normalizes (configurable lowercase/stopwords), and lemmatizes. Return list of tokens

### Success criteria
- [ ] Students can preprocess any text passage
- [ ] Students can explain tradeoffs (lowercase, stopwords, stem vs lemma)

### Health data parallel
Same pipeline applies to clinical notes: tokenization affects "500mg" vs "500"+"mg"; stopword removal must preserve "no"/"not" for negation.

---

## Demo 2: Literary Detective Work

**File:** `02a_analysis.md` → `02a_analysis.ipynb`
**Lecture sections:** Part-of-Speech Tagging, Named Entity Recognition, Text Extraction (Regex)
**Timing:** After POS, NER, and Text Extraction (second demo break)
**Text:** Sherlock Holmes, "A Scandal in Bohemia" through King's explanation (~4,000 words)

### Goal
- Extract structured information from unstructured narrative
- Use both NLTK and spaCy for POS and NER
- Extract dates, addresses, times with regex
- Combine into a "cast list" and timeline

### Walkthrough

1. **Load Holmes excerpt:** Read `data/holmes_scandal.txt`, note rich entities (names, places, dates, times)
2. **POS tag (NLTK):** `nltk.word_tokenize` + `nltk.pos_tag`. Extract nouns: `[w for w,t in tagged if t.startswith('NN')]` — characters, objects, locations
3. **POS tag (spaCy):** `nlp(text)`, iterate tokens with `token.pos_`, `token.tag_`. Compare tagsets (Penn Treebank vs Universal Dependencies)
4. **NER (NLTK):** `nltk.ne_chunk(tagged)`, iterate chunks with `label()`. Note tree structure
5. **NER (spaCy):** `doc.ents` — iterate with `ent.text`, `ent.label_`. Expected: Holmes, Watson, King, Irene Adler, Baker Street, Bohemia, dates
6. **Regex: dates** — pattern for "March 20th, 1888" style; numeric dates `\d{1,2}/\d{1,2}/\d{4}`
7. **Regex: addresses** — e.g. "221B Baker Street" (word + number + words)
8. **Regex: times** — "quarter past eleven", "half-past"
9. **Combine:** Build character tracker (who appears) and simple timeline from extracted dates/times

### Discussion prompt
"What entities did NER miss?" (e.g. "the King" not tagged as PERSON.) Same limitation in clinical text ("the patient" not tagged).

### Success criteria
- [ ] Students extract a "cast list" (people, places) and timeline of events
- [ ] Students can explain NER vs regex: semantic entities vs syntactic patterns

### Health data parallel
EHR extraction: person names, organization (hospital), dates (DOB, visit dates), addresses, medication timing — same toolbox.

---

## Demo 3: Comparing Literary Worlds

**File:** `03a_comparison.md` → `03a_comparison.ipynb`
**Lecture sections:** Bag of Words, TF-IDF, N-grams, Document Similarity, Pipelines
**Timing:** End of lecture (third demo break)
**Texts:** All four excerpts as mini corpus (Alice, Holmes, Pride and Prejudice, Frankenstein)

### Goal
- Vectorize documents and compare them quantitatively
- See TF-IDF distinctive terms per text
- Compute pairwise cosine similarity and visualize (heatmap)
- Build complete pipeline (NLTK and spaCy) and compare

### Walkthrough

1. **Load corpus:** Read all four files from `data/` into a list of strings (or dict with labels)
2. **Bag of Words:** `CountVectorizer()`, `fit_transform(docs)`, inspect `get_feature_names_out()`, `X.toarray()` — document-term matrix
3. **TF-IDF:** `TfidfVectorizer()`, fit and transform. For each document, show top distinctive terms (highest TF-IDF). Expect "Watson" for Holmes, "creature" for Frankenstein, etc.
4. **N-grams:** `CountVectorizer(ngram_range=(1,2))`, fit. Show bigrams: "my dear Watson", "Mr. Darcy", "chest pain" style phrases
5. **Cosine similarity:** `cosine_similarity(X)` on TF-IDF matrix. Print pairwise matrix
6. **Visualize:** Heatmap of similarity matrix (seaborn or matplotlib), label axes with text names
7. **NLTK pipeline:** Implement function that tokenizes, normalizes, lemmatizes, POS-tags, NER-chunks, returns dict (tokens, nouns, entities). Use lecture snippet as template
8. **spaCy pipeline:** Implement function that runs `nlp(text)` and returns dict (tokens, lemmas, nouns, entities). One call does all
9. **Compare:** When to use manual (NLTK) vs integrated (spaCy)? Reference lecture "Comparing the Approaches" table

### Mini-challenge
Students add a fifth text (their choice from Gutenberg or elsewhere), append to corpus, re-run TF-IDF and similarity — where does it cluster?

### Success criteria
- [ ] Students can vectorize a corpus and compute document similarity
- [ ] Students can build and run both NLTK and spaCy pipeline functions

### Health data parallel
Comparing documents = same whether novels or discharge summaries; distinctive terms and similarity drive search, clustering, and cohort discovery.

---

## Converting Markdown to Notebooks

Before class:

```bash
cd lectures/04/demo
pip install jupytext   # if needed
jupytext --to notebook 01a_preprocessing.md
jupytext --to notebook 02a_analysis.md
jupytext --to notebook 03a_comparison.md
```

Keep `.md` files as source of truth; regenerate `.ipynb` as needed.

---

## Environment

- **requirements.txt:** nltk, pyyaml, spacy, scikit-learn, jupytext, and the spaCy English model wheel (one-shot install: `uv pip install -r requirements.txt` or `pip install -r requirements.txt` from this directory).
- **config.yaml:** Data directory (`data.dir`), corpus files (`data.files`), spaCy model (`spacy.model`), and visualization defaults (`visualization.top_tokens`, `visualization.top_tfidf`). Demos and `prepare_data.py` read from it.
- NLTK data (punkt_tab, stopwords, wordnet, etc.) is downloaded in the first cell of each demo.
