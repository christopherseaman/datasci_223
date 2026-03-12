# Lecture 04 Notes — Natural Language Processing

# Links & Self-Guided Review

Jurafsky & Martin's *Speech and Language Processing* is the gold-standard NLP textbook, free online. Chapters 2-6 map directly to this lecture. For students who want one post-lecture resource, the spaCy 101 guide is the most efficient — it covers the entire object model in one page with runnable examples.

Regex101 is worth bookmarking: the real-time explanation panel teaches regex faster than any tutorial because you get instant feedback on what each piece of a pattern does.

The scispaCy link matters for anyone working with biomedical or clinical text for projects — it provides pre-trained models that recognize genes, chemicals, diseases, and cell types that standard spaCy misses entirely.

# Natural Language Processing

## What is NLP?

The fundamental tension in NLP is that human language evolved for human brains, not computers. We communicate with ambiguity *on purpose* — it's a feature, not a bug. When you say "I saw her duck," a human uses context (is she at a pond? at a bar?) to resolve the meaning instantly. A computer sees a subject, a verb, and then... is "duck" a noun or a verb? Is "her" possessive or objective?

Language is also deeply *redundant*. In English, roughly 50% of characters in a sentence are predictable from context alone (Shannon's entropy experiments from the 1950s showed this). NLP exploits that redundancy — statistical patterns in how words co-occur let algorithms make surprisingly good guesses about meaning without truly "understanding" anything.

The clinical context makes this harder. A doctor writing "pt c/o SOB" is using a compression scheme (abbreviations) that saves time but loses the redundancy NLP algorithms rely on. Clinical text is the hardest NLP domain because it combines extreme abbreviation, domain-specific jargon, critical negation ("denies CP"), and life-or-death stakes for getting it wrong.

The four core challenges — ambiguity, context, variation, implicit knowledge — are each worth a quick concrete example. "Bank" is the classic ambiguity case. "Not bad" shows how context inverts literal meaning. The variation example ("BP" / "blood pressure" / "b.p.") is one students will encounter immediately in EHR data. "Take with food" requiring the reader to know what meals are illustrates how much common-sense knowledge humans bring to language that machines lack.

## Why NLP for Health Data?

About 80% of clinical data is unstructured text — progress notes, radiology reports, discharge summaries. ICD codes and structured fields capture the billing story, but the clinical story lives in free text. A patient's chart might have an ICD code for "chest pain" but the nuance — was it exertional? substernal? relieved by rest? — is in the note.

Practical example: identifying patients who experienced an adverse drug reaction. Structured data might show the drug was prescribed and later discontinued, but the *reason* for discontinuation (rash? GI upset? patient preference?) is buried in a note. NLP lets you systematically extract that reason across thousands of charts.

Another concrete use: cohort building for clinical trials. "Find all patients with treatment-resistant depression" requires reading notes to determine that multiple medications were tried and failed — something that structured diagnosis codes alone can't reliably capture.

Each clinical data source maps to a different NLP sub-task. Progress notes like "Patient denies chest pain, reports mild fatigue" require NER *plus* negation detection — you need to know "chest pain" is negated and "fatigue" is affirmed. Radiology reports like "No acute intracranial abnormality" require understanding that the entire finding is negative. Discharge summaries need temporal extraction ("follow up in 2 weeks"). Patient survey feedback like "The wait time was frustrating" is sentiment classification. NLP isn't one technique — it's a toolbox, and the right tool depends on what you're extracting from which source.

## Classical vs LLM-based Approaches

Classical NLP is *glass box* engineering — you can open it up, see every gear, and explain exactly why the system made a decision. When a classical pipeline says "this note mentions chest pain," you can trace it: the tokenizer split the text, the TF-IDF vectorizer weighted "chest" and "pain," and the classifier threshold was 0.7. That interpretability matters for clinical applications where you need to explain decisions to clinicians, IRBs, and regulators.

LLMs work more like a black box that happens to be extremely good at pattern recognition. GPT-style models learn contextual representations where the same word gets different vectors depending on surrounding words — "bank" near "river" vs "bank" near "account" get different internal representations. That contextual understanding is powerful but opaque.

The practical tradeoff: classical NLP is cheap, fast, interpretable, and works with small labeled datasets. LLMs are expensive, slow, opaque, but handle nuance and generalization much better. Most production clinical NLP systems today use classical techniques for structured extraction (pulling out vitals, medications, dates) and reserve LLMs for tasks requiring deeper understanding (summarization, question answering, complex reasoning about patient trajectories).

The representation difference is key to understanding the tradeoff. Classical NLP represents text as sparse vectors where each dimension is a known word — you can inspect dimension 47 and see it means "diabetes." LLM embeddings represent text as dense vectors (768+ dimensions) where no single dimension has human-interpretable meaning — the meaning is distributed across all of them. That opacity is the price of the richer representation, and it's why classical methods remain valuable when you need to explain your model to a clinician or regulator.

## Tools: NLTK vs spaCy

NLTK was built by academics for academics. Steven Bird and Edward Loper created it in 2001 at the University of Pennsylvania as a teaching tool. It exposes the internals — you can swap in different tokenizers, stemmers, taggers, and see how each algorithm behaves differently on the same input. That transparency is why we use it in class.

spaCy was built by Matthew Honnibal (a computational linguist) starting in 2015 specifically for production use. Instead of offering five tokenizers and letting you choose, spaCy gives you one that's been optimized for speed and accuracy. The object model (`Doc` → `Token` → `Span`) means you process text once and every annotation (POS, NER, dependency parse, lemma) is immediately available on the same objects.

The processing model difference matters: NLTK works on strings and lists, so you're constantly converting between representations. spaCy processes everything into a `Doc` object in one pass — the pipeline runs tokenization, tagging, parsing, and NER sequentially, and each component annotates the same data structure.

The `Doc` / `Token` / `Span` trio is spaCy's core abstraction. `Doc` is the whole processed document. `Token` is a single word with all its annotations (text, POS, lemma, whether it's a stopword). `Span` is a contiguous slice of tokens — entities are Spans, noun chunks are Spans. Everything composes from these three objects.

### Installation

Both libraries require a two-step install: the library itself, then data/models. NLTK needs `nltk.download('punkt_tab')`, `nltk.download('stopwords')`, etc. — each task uses a separate downloadable data package. spaCy needs `python -m spacy download en_core_web_sm` to fetch the English pipeline model. The `_sm` suffix means "small" — fast but no word vectors. `_md` (medium) and `_lg` (large) include pre-trained word vectors but are much bigger downloads.

Students will hit this in the first demo: forgetting `nltk.download()` calls gives `LookupError` messages; forgetting the spaCy model download gives `OSError: Can't find model 'en_core_web_sm'`. Both are common first-time setup issues and nothing is broken — just missing data.

# Text Processing Fundamentals

This section covers the preprocessing pipeline that converts raw text into a consistent form suitable for analysis. Every NLP system, classical or modern, starts with some version of these steps. The order matters: tokenize first, then normalize, then reduce words to base forms. Each step is lossy — it discards some information in exchange for regularity — and the right choices depend entirely on the downstream task.

Think of it as an assembly line: raw text goes in one end, a clean list of normalized tokens comes out the other. Decisions at each station propagate forward — a bad tokenization choice poisons every subsequent step.

## Tokenization

Tokenization seems trivial until you try it. The sentence "Dr. Smith prescribed 500mg ibuprofen" has at least three reasonable tokenizations depending on your downstream task:

- If you're counting word frequencies, you probably want "500mg" as one token
- If you're extracting dosages, you need "500" and "mg" separated
- If you're doing NER, you need "Dr." to stay together (it's a title, not a sentence-ending period)

Each tokenizer makes different engineering choices. Whitespace split is the naive baseline — it keeps punctuation attached ("ibuprofen.") and can't handle abbreviations. NLTK's `word_tokenize` uses a trained model (Punkt) that knows "Dr." is an abbreviation, not a sentence end, and splits final punctuation from words. spaCy goes further: it has explicit rules for numbers followed by units, so "500mg" becomes ["500", "mg"] — useful for dosage extraction but counterproductive if you want to count "500mg" as a single concept. No tokenizer is objectively "correct"; each is optimized for different downstream tasks.

Sentence tokenization is often overlooked but equally important. `nltk.sent_tokenize()` uses a pre-trained model (Punkt) that knows "Dr." isn't a sentence boundary but "daily." is. Naive splitting on periods would break on abbreviations, decimal numbers, and URLs.

Subword tokenization (used by modern LLMs) takes a different approach entirely. Instead of splitting on whitespace and punctuation, BPE (Byte Pair Encoding) builds a vocabulary from character pairs that frequently co-occur. So "tokenization" might become "token" + "ization" — the model learns morphological structure implicitly. This elegantly handles rare words and typos: even if "cardiomyopathy" isn't in the vocabulary, its subpieces probably are.

The critical point: your tokenizer and your model must match. You can't tokenize with NLTK's word tokenizer and feed the result to a BERT model that expects WordPiece tokens. The token boundaries would be wrong and every downstream prediction would be garbage.

## Normalization

Lowercasing is a lossy compression. You're trading information (case) for simplicity (smaller vocabulary). In general English, that's usually fine — "The" and "the" are the same word. But clinical text is case-sensitive in ways that matter: "MS" (multiple sclerosis) vs "ms" (milliseconds), "US" (ultrasound) vs "us" (pronoun), "ED" (emergency department) vs "ed" (past tense suffix).

Stopword removal is similarly context-dependent. NLTK's English stopword list has 179 words. For topic modeling or document similarity, removing them focuses the signal on content words. But for clinical NLP, negation words ("no," "not," "without," "denies") are stopwords in the NLTK list that carry critical clinical meaning. "No chest pain" with stopwords removed becomes "chest pain" — the opposite meaning.

Punctuation removal is the third normalization step. Usually safe to strip commas and periods, but hyphens in compound terms ("COVID-19," "follow-up") and slashes in vitals ("120/80") carry meaning. A blanket `string.punctuation` removal destroys those — in production you'd selectively remove some punctuation while preserving diagnostically meaningful characters.

The general principle: preprocessing is not neutral. Every transformation you apply is a modeling decision that assumes certain information is irrelevant. The right preprocessing depends entirely on what you're trying to do downstream.

## Stemming and Lemmatization

Stemming is a heuristic: chop off suffixes according to rules. Porter's stemmer (1980) uses about 60 rules like "if the word ends in -ies and has more than one syllable, replace with -i." Fast, simple, language-specific. The downside: "university" → "univers," "studies" → "studi" — these aren't real words, so you can't look them up in a dictionary or display them to a human.

Lemmatization is a lookup: find the dictionary headword. "better" → "good" (if you know it's an adjective), "ran" → "run" (if you know it's a verb). This requires knowing the part of speech — "meeting" could lemmatize to "meeting" (noun) or "meet" (verb). spaCy handles this automatically because its pipeline runs POS tagging before lemmatization.

The differences become clear on irregular words. "Running" → both produce "run" (the suffix rule happens to match the dictionary form). But "studies" → "studi" (stemmer applies -ies → -i rule, producing a non-word) vs "study" (lemmatizer looks up the dictionary form). "Better" is the most instructive case: the stemmer leaves it unchanged because no suffix-stripping rule fires, while the lemmatizer maps it to "good" — but only if it knows "better" is an adjective. Without POS information, a lemmatizer can't distinguish "better" (adjective → "good") from "better" (verb, as in "to better oneself" → "better").

When to use which: stemming is fine when you just need to group related words and don't care about readability (e.g., building features for a classifier). Lemmatization is better when you need to display results to humans or when exact word forms matter (e.g., matching against a medical terminology database like UMLS).

# LIVE DEMO!

Demo 1 (Text Preprocessing Safari) uses Alice in Wonderland, Chapter 1. The literary text has useful edge cases that parallel clinical challenges: "didn't" tests contraction tokenization (does it become ["did", "n't"] or ["didn't"]?), "rabbit-hole" tests hyphenated compound handling, and "curiouser" is a non-standard word that stemmers and lemmatizers handle differently (Porter stem: "curious"; lemma: "curiouser" — no dictionary entry). The health data parallel: same decisions apply to clinical notes — tokenization affects "500mg," stopword removal must preserve negation words like "no" and "not." Students build a reusable `preprocess(text)` function that tokenizes, normalizes, and lemmatizes with configurable options.

# Part-of-Speech Tagging

## Concepts

POS tagging is one of those NLP tasks that seems academic until you need it. Two practical uses:

1. **Disambiguation for lemmatization**: "running" as a verb lemmatizes to "run"; as a noun ("running of the bulls") it stays "running." The POS tag tells the lemmatizer which form to produce.

2. **Information extraction**: If you want to find the main topics in a clinical note, extracting nouns gives you a useful first pass. Adjective + noun patterns ("severe pain," "chronic fatigue," "acute onset") capture clinically meaningful descriptors.

Consider the sentence "The patient reported severe chest pain yesterday." A human parses this instantly: "the" modifies "patient" (DT → NN), "severe" modifies "pain" (JJ → NN), "reported" is the action (VBD), "yesterday" is temporal context (RB). POS tagging makes this implicit grammatical structure explicit and machine-readable — every token gets a label that downstream tools can filter, group, or count.

### Common POS Tags

Penn Treebank tags (used by NLTK) are fine-grained: NN (singular noun), NNS (plural noun), NNP (proper noun singular), NNPS (proper noun plural), VB (verb base form), VBD (past tense), VBG (gerund), JJ (adjective), RB (adverb), DT (determiner). Universal Dependencies tags (used by spaCy) are coarser: all noun variants map to just "NOUN" or "PROPN." The fine-grained tags give you more information; the universal tags are easier to work with and consistent across languages.

The most useful distinction for beginners: NN-family = nouns (the "things"), VB-family = verbs (the "actions"), JJ = adjectives (the "descriptors"), RB = adverbs (modifies verbs/adjectives). For clinical text extraction, NN + JJ combinations capture most clinically meaningful content — "severe pain," "chronic condition," "bilateral edema."

Modern POS taggers achieve ~97% accuracy on well-edited English text. On clinical text, accuracy drops to ~93-95% because of sentence fragments, abbreviations, and non-standard grammar ("pt a&o x3, no acute distress").

## NLTK

NLTK's POS tagging is a two-step process: tokenize with `word_tokenize()`, then tag with `pos_tag()`. The output is a list of `(word, tag)` tuples. This separation means you can use any tokenizer before tagging — flexibility that matters when exploring different preprocessing strategies.

The basic extraction pattern is: tag everything, then filter by tag. A list comprehension like `[word for word, tag in tagged if tag.startswith('NN')]` pulls all nouns from a tagged sentence. Starting the match with "NN" catches singular (NN), plural (NNS), proper singular (NNP), and proper plural (NNPS) in one pass. Swap the prefix to "VB" for verbs, "JJ" for adjectives.

Nobody memorizes the full Penn Treebank tagset. `nltk.help.upenn_tagset('NN')` prints a definition and examples for any tag on the fly — that's the workflow.

## spaCy

spaCy provides both coarse (`token.pos_`) and fine-grained (`token.tag_`) POS tags on every token automatically — no separate tokenize-then-tag step. One `nlp(text)` call does it all.

`spacy.explain(tag)` is the spaCy equivalent of NLTK's help function. Pass in any tag string and get a human-readable description.

spaCy exposes both tag systems on the same token: `token.pos_` gives the coarse Universal Dependencies tag (DET, NOUN, VERB — language-agnostic, works across English/Spanish/German) and `token.tag_` gives the fine-grained Penn Treebank tag (DT, NN, VBD — English-specific, distinguishes singular from plural from proper nouns). Use coarse tags for simple filtering, fine tags when the distinction matters (e.g., proper nouns only for NER preprocessing).

# Named Entity Recognition

## Concepts

NER is pattern recognition for proper nouns and structured concepts. Standard NER models recognize a fixed set of entity types — PERSON, ORG, GPE (geopolitical entity), DATE, MONEY, etc. These categories were defined by the Message Understanding Conferences (MUC) in the 1990s for news text.

A single clinical sentence like "Dr. Smith at UCSF prescribed Metformin 500mg on January 15" contains five entity types at once: a person (Dr. Smith), an organization (UCSF), a medication (Metformin), a quantity (500mg), and a date (January 15). Standard NER catches three of those — PERSON, ORG, DATE — because they're in the standard entity set inherited from news-text corpora. But "Metformin" gets missed entirely: medications aren't a standard NER category.

Clinical text needs entity types that general models don't support: MEDICATION, DOSAGE, DIAGNOSIS, PROCEDURE, ANATOMY. Recognizing these requires models trained on annotated clinical corpora — that's what scispaCy and MedSpaCy provide.

A key limitation: NER works on *mentions*, not *resolution*. If a note says "Dr. Smith" and later "the attending physician," NER identifies both as entities but doesn't know they're the same person. That's a separate task called coreference resolution, which is much harder.

## NLTK

NLTK's NER requires POS-tagged input — tokenize, POS-tag, then pass tagged tokens to `ne_chunk()`. Three explicit steps reflecting NLTK's manual-pipeline philosophy.

NLTK returns an `nltk.Tree` because named entities can span multiple tokens — "New York City" is one GPE, not three separate words. In the tree, multi-token entities become subtrees (with a `.label()` method returning the entity type), while non-entity tokens stay as leaves. To extract entities, you iterate the tree and check `hasattr(chunk, 'label')` to distinguish entity subtrees from plain-token leaves. It's awkward compared to spaCy's approach, but it makes the underlying structure explicit.

The `binary=True` option simplifies to just NE vs non-entity — useful when you care about *where* entities are but not *what type* they are.

NLTK's NER is noticeably less accurate than spaCy's out of the box — it uses an older MaxEnt chunker, while spaCy uses a neural model trained on OntoNotes. Running both on the same text makes the gap visible: spaCy reliably tags multi-word names and organizations that NLTK misses or misclassifies.

## spaCy

spaCy's NER is accessed through `doc.ents` — a tuple of `Span` objects. Each entity has `.text` (the surface string), `.label_` (entity type), `.start` and `.end` (token indices). Much cleaner than NLTK's tree structure.

spaCy's NER is better out of the box because its model was trained on OntoNotes, a large annotated corpus, using a neural transition-based parser. It reliably catches PERSON, ORG, DATE in well-formed text. It still struggles with clinical-specific entities not in its training data.

spaCy entity extraction is concise: `for ent in doc.ents: print(ent.text, ent.label_)` — two lines to get every entity and its type. NLTK requires tree-walking with `hasattr` checks. Same information, but the API difference reflects the design philosophy: spaCy optimizes for the common case (just give me the entities), NLTK exposes the underlying data structure (here's the parse tree, extract what you need).

# Text Extraction

This bridges NER and regex. NER finds *semantic* entities (this is a person, this is an organization). But many clinical extraction tasks are *syntactic* — you're looking for patterns in how characters are arranged, not what they mean. Blood pressure is always "two-to-three digits, slash, two-to-three digits." Dosages are "number followed by unit." These structural patterns are regex territory.

## Regex Patterns

Regular expressions are the Swiss army knife of text extraction. NER finds semantic entities (this is a person, this is a date). Regex finds structural patterns (three digits, a slash, two digits).

For clinical text, regex is indispensable for:
- **Vitals**: `\d{2,3}/\d{2,3}` matches blood pressure readings like "120/80"
- **Dosages**: `\d+\s?(mg|ml|mcg|units)` matches "500mg", "10 ml", "100 units"
- **Lab values**: `\d+\.?\d*\s?(mg/dL|mmol/L|mEq/L)` matches lab results with units
- **Dates in various formats**: clinical systems use every date format imaginable

The `re` module has three workhorses: `re.search()` finds the first match (use when you just need to know if a pattern exists), `re.findall()` finds all matches (use when you need every occurrence), `re.sub()` replaces matches (use for redaction, normalization, or cleaning). Capture groups `(...)` let you extract specific parts of a match — the medication name, dose number, and unit separately.

The power of regex is composability — you can build complex patterns from simple pieces. The danger is fragility — a regex that matches "500mg" won't match "500 mg" (with a space) unless you explicitly account for optional whitespace with `\s?`.

Regex101.com is a genuinely useful tool for building and testing patterns interactively. The explanation panel shows exactly what each part of the pattern does, which is invaluable when debugging.

## Regex Across NLP Tools

Regex isn't confined to the `re` module — it shows up throughout the Python data science stack. Learning regex once pays off in many contexts:

- **NLTK's `RegexpTokenizer`** uses regex to define what constitutes a token. `r'\w+'` means "one or more word characters" — tokenize while stripping punctuation in one step, rather than tokenizing and then filtering.
- **Pandas `.str` methods** accept regex for operations on text columns. `df['notes'].str.extract(r'(\d+)/\d+')` pulls systolic blood pressure from a column of vitals strings — that's how regex scales from one note to a dataframe of thousands.
- **spaCy's `Matcher`** uses token-level patterns rather than character-level regex, but the concept is analogous: define a structural pattern, find all matches in processed text. For example, you could match blood pressure readings by token shape — `{"SHAPE": "ddd"}`, `{"TEXT": "/"}`, `{"SHAPE": "dd"}` matches "120/80" using spaCy's token attributes rather than raw character patterns.

The return on learning regex is high because the syntax is nearly identical across Python, R, SQL, and command-line tools like `grep`.

# LIVE DEMO!!

Demo 2 (Literary Detective Work) uses Sherlock Holmes text to exercise POS tagging, NER, and regex together. The text is rich in extractable entities: character names (Holmes, Watson, the King, Irene Adler), locations (Baker Street, Bohemia), dates, times. Students build a "cast list" and timeline — a concrete information extraction task that directly parallels clinical chart extraction (who, where, when, what happened).

Key discussion point: "What entities did NER miss?" Standard NER expects proper names — "Sherlock Holmes" gets tagged, but "the King" likely doesn't because it's a title, not a name. This is the same limitation in clinical text: "the patient" isn't tagged as a person entity, "the attending" isn't tagged as a role. NER finds explicit named references, not implicit or role-based ones.

# Text Representation

Up to now we've been processing text — cleaning, tagging, extracting. This section shifts to *representing* text as numbers so it can be input to machine learning algorithms. The fundamental question: how do you convert words into vectors that a classifier or clustering algorithm can work with?

There's a spectrum from simple (Bag of Words) to sophisticated (word embeddings), trading interpretability for expressiveness at each step.

## Bag of Words

Bag of Words is the simplest text-to-numbers conversion: count how many times each word appears in each document. The result is a document-term matrix — rows are documents, columns are words, values are counts.

The fundamental assumption: word order doesn't matter. "Patient denies pain" and "Pain denies patient" produce identical vectors. This is obviously wrong linguistically, but surprisingly useful in practice for tasks like document classification and clustering. The statistical distribution of words is often enough to distinguish document types.

Consider three short clinical notes: "patient reports chest pain," "patient denies chest pain," "patient reports headache." The vocabulary is six words; the document-term matrix is 3x6. Each cell is a count of how many times that word appears in that document. Documents 1 and 2 are nearly identical vectors (both contain "patient," "chest," "pain") despite meaning opposite things — BoW can't distinguish "reports" from "denies" as semantically opposed because it treats all words as independent dimensions.

In real corpora the vocabulary runs to thousands or tens of thousands of unique words. The resulting matrix is enormously sparse — any given document uses a tiny fraction of the total vocabulary, so most cells are zero. scikit-learn's `CountVectorizer` handles the full pipeline (tokenize, build vocabulary, count) in one object: `fit_transform()` learns and transforms, `get_feature_names_out()` maps column indices back to words.

Three limitations of BoW each motivate a subsequent technique: word order is lost (N-grams partially recover it); common words dominate the counts (TF-IDF downweights them); every word is equally distant from every other word (word vectors encode semantic similarity).

## TF-IDF

TF-IDF adds one important insight: words that appear in every document aren't useful for distinguishing between documents. "Patient" appears in every clinical note — high term frequency, low discriminative value. "Metformin" appears in only diabetes-related notes — that's informative. IDF (inverse document frequency) mathematically downweights ubiquitous words and upweights distinctive ones.

The formula: TF-IDF = TF x IDF, where IDF = log(total documents / documents containing the word). A word in every document gets IDF = log(1) = 0, zeroing out its contribution entirely. A word in only one document gets the maximum IDF weight.

scikit-learn's `TfidfVectorizer` uses a smoothed IDF formula: log((n+1)/(df+1)) + 1, where n is total documents and df is documents containing the word. The smoothing (adding 1 to numerator and denominator) prevents division by zero and keeps rare words from dominating too aggressively. If students hand-calculate IDF and compare to scikit-learn output, the numbers will differ slightly because of this smoothing.

Concretely: in a 3-document corpus, "patient" (in all 3) gets IDF = 1.00 while "denies" (in only 1) gets IDF = 1.69. TF-IDF is doing automatic feature selection — ubiquitous words get suppressed, distinctive words get amplified, without maintaining an explicit stopword list.

Key API distinction: `fit_transform()` for training data (learns vocabulary + transforms), `transform()` for new data (uses existing vocabulary, no refitting). Calling `fit_transform()` on test data is a subtle bug — you'll get a different vocabulary and the model won't work.

## N-grams

Single words (unigrams) lose context that matters. "Chest pain" as a bigram is a clinical concept; "chest" and "pain" separately are just body parts and symptoms. "No chest pain" as a trigram captures negation; the unigram "pain" doesn't.

N-grams are a simple way to capture local word order without building a full syntax model. The tradeoff is vocabulary explosion: if you have V unique words, you can have up to V^2 unique bigrams and V^3 unique trigrams. In practice, most possible n-grams never occur, so the matrix is extremely sparse.

The `ngram_range=(1, 2)` parameter in scikit-learn's vectorizers means "include both unigrams and bigrams." This is a common practical choice — you get individual word signals plus the most important two-word phrases. Going to trigrams `(1, 3)` captures more context but the feature space explodes.

For clinical text, bigrams and trigrams are particularly valuable because so much clinical language is multi-word: "chest pain," "shortness of breath," "blood pressure," "heart rate," "white blood cell count."

N-grams aren't a separate tool — they're a parameter on the same `CountVectorizer` or `TfidfVectorizer` students already know.

## Word Vectors

Bag of Words and TF-IDF treat every word as equally different from every other word. "Diabetes" and "hypertension" are just as distant as "diabetes" and "pizza." Word vectors (embeddings) fix this by mapping words into a continuous space where semantic similarity corresponds to geometric proximity.

The intuition: words that appear in similar contexts should have similar meanings. "Diabetes" and "hypertension" both appear near words like "patient," "management," "chronic," "medication." "Pizza" appears near "delivery," "toppings," "cheese." The embedding algorithm learns these co-occurrence patterns and encodes them as dense vectors (typically 100-300 dimensions).

spaCy's small model (`en_core_web_sm`) doesn't include word vectors — it's optimized for pipeline speed. The medium (`_md`) and large (`_lg`) models include 300-dimensional GloVe vectors. Each token gets a `.vector` attribute (a 300-element numpy array), and `.similarity()` computes cosine similarity between any two tokens or spans. "Diabetes" and "hypertension" might score ~0.5 similarity; "diabetes" and "pizza" near 0.

This is a conceptual bridge to embeddings in depth later. The key takeaway: BoW/TF-IDF vectors are *sparse* and *interpretable* (each dimension is a known word, most are zero), word vectors are *dense* and *semantic* (all 300 dimensions are nonzero, none corresponds to a single word, but similar words cluster together in the space).

# Document Similarity

With text represented as vectors, we can measure how similar documents are. This is the payoff of all the representation work — quantitative comparison enables search, clustering, recommendation, and deduplication.

## Cosine Similarity

Cosine similarity measures the angle between two vectors, not the distance. This is important because document length affects vector magnitude but not direction. A 10-page discharge summary and a 1-paragraph progress note about the same condition will have very different vector magnitudes (because the longer document has higher word counts) but similar directions (because the relative proportions of words are similar).

The formula — dot product divided by the product of magnitudes — normalizes for length. Cosine similarity ranges from 0 to 1 for TF-IDF vectors (because all values are non-negative). A score of 1.0 means the documents use exactly the same words in exactly the same proportions. A score of 0.0 means they share no words at all. In practice, even very similar documents rarely exceed 0.5-0.7 because natural language variation introduces many unique word choices.

A concrete example: compare three clinical snippets — "chest pain and shortness of breath," "chest discomfort and difficulty breathing," and "headache and nausea." The first two share symptom concepts (chest, breathing) even though they use different words, so cosine similarity is moderate (~0.35). The third shares almost nothing with either (~0.11). The resulting similarity matrix is always symmetric (sim(A,B) = sim(B,A)) with 1.0 on the diagonal (every document is identical to itself).

Jaccard similarity is a simpler alternative: |intersection| / |union| of word *sets*, ignoring frequency. A word either appears or doesn't. It's faster and more intuitive but loses the frequency weighting that makes TF-IDF powerful. Computing both on the same documents makes the tradeoff concrete — Jaccard treats "chest" appearing once the same as "chest" appearing ten times.

Practical applications in health data: finding similar patient notes (for case-based reasoning), deduplicating records, building search systems over clinical text, clustering discharge summaries by condition type.

# Specialized Tools

General NLP models are trained on news, web text, and Wikipedia. Clinical text is different enough that these models underperform. This section surveys tools built specifically for biomedical and clinical language.

**scispaCy** provides models trained on biomedical literature (PubMed abstracts, PMC full texts). Same spaCy API, different training data — drop-in replacement that recognizes genes, chemicals, diseases, and cell types that general models miss.

**MedSpaCy** adds clinical-specific pipeline components, most importantly negation detection. The NegEx algorithm (Chapman et al., 2001) is a simple but effective rule-based approach: it looks for negation triggers ("no," "denies," "without," "ruled out") and checks whether they're within a window of a clinical finding. "No chest pain" → chest pain is negated. "History of MI, no recurrence" → MI is affirmed, recurrence is negated.

**cTAKES** (clinical Text Analysis and Knowledge Extraction System) is a Java-based clinical NLP pipeline from the Apache Foundation. It handles tokenization through NER, negation, and temporal resolution with models trained on clinical text. Heavyweight but comprehensive.

**MetaMap** from the National Library of Medicine maps free text to UMLS concepts — "heart attack" → C0027051 (myocardial infarction). Requires an NLM license, computationally slow, but nothing matches its coverage for concept normalization.

**UMLS** (Unified Medical Language System) is the Rosetta Stone of medical terminologies. It maps between ICD codes, SNOMED CT, RxNorm, MeSH, and dozens of other coding systems. When NLP extracts "heart attack" from text, UMLS can map that to the SNOMED concept for "myocardial infarction" (22298006), which links to the ICD-10 code I21. This concept normalization is essential for combining NLP-extracted information with structured clinical data.

# Challenges

Five challenges that make clinical NLP substantially harder than general-domain NLP:

**Abbreviations** — "pt c/o SOB" = "patient complains of shortness of breath." A study of clinical abbreviations found that the average abbreviation has 3.5 possible meanings. "MS" alone has at least 12 medical meanings (multiple sclerosis, mental status, morphine sulfate, mitral stenosis...). Context usually disambiguates for a human reader, but automatic disambiguation requires models trained on clinical text. Some institutions maintain abbreviation dictionaries, but usage varies across departments and individual clinicians.

**Negation** is the most critical. A study by Chapman et al. found that up to 50% of clinical findings mentioned in radiology reports are negated. Simple keyword extraction without negation detection will have an unacceptably high false positive rate. "No evidence of malignancy" is a *good* finding, not a cancer diagnosis.

A minimal regex negation detector like `re.search(r'\b(no|denies?|without)\s+\w+\s+chest\s+pain', text, re.I)` catches simple cases: "no chest pain," "denies any chest pain." But it breaks on complex sentence structure: "No evidence of the previously reported chest pain" has too many words between the trigger ("no") and the finding ("chest pain") for the pattern to match. Real negation detection algorithms (NegEx, ConText) define explicit scope rules — a negation trigger negates everything within its forward scope until a scope-breaking word (like a period, "but," or "however") is encountered.

**Uncertainty** is clinically important but linguistically subtle. "Possible pneumonia" vs "pneumonia" vs "no pneumonia" represent three different clinical states. Hedge words ("possible," "likely," "suspected," "cannot rule out") are common in clinical notes because physicians often document differential diagnoses, not definitive ones. "Rule out MI" means suspicion, not confirmation — a system that treats it as a positive finding will generate false alerts.

**Temporality** distinguishes current conditions from history. "History of breast cancer" (past, possibly resolved) vs "breast cancer" (current) vs "family history of breast cancer" (not the patient's condition at all) require temporal and relational reasoning that basic NLP pipelines don't handle. The word "history" alone completely changes the clinical interpretation.

**Variations** — "hypertention" (misspelling), "htn," "HTN," "high blood pressure" all mean the same clinical concept. Lowercasing handles some variation ("HTN" = "htn"), but abbreviation expansion and spelling correction require additional tools or terminology dictionaries. In a large health system, the same condition may be documented dozens of different ways across clinicians.

# Pipelines

This section shows how the individual techniques (tokenization, normalization, lemmatization, POS tagging, NER, regex) combine into end-to-end processing pipelines. The contrast between NLTK and spaCy pipelines makes a concrete point about engineering tradeoffs: manual assembly gives control, integrated systems give speed.

## NLTK Pipeline

The NLTK pipeline is like cooking from scratch — you choose every ingredient and control every step. Tokenize, then filter stopwords, then lemmatize, then POS tag, then NER, then regex. If you want to change one step (different stopword list, different stemmer), you just swap that component. The cost is boilerplate code and slower execution.

A typical NLTK pipeline function takes raw text and returns a structured dictionary: tokens, POS tags, nouns, entities, and regex-extracted patterns. Each transformation is an explicit line of code. This explicitness enables domain-specific customization — for example, modifying the stopword set with `stop_words = set(stopwords.words('english')) - {'no', 'not'}` to preserve negation words that are critical in clinical context. That kind of targeted override is natural in a manual pipeline.

On a clinical note like "Patient John Smith, age 45, presents with BP 140/90 and chest pain," the pipeline produces a traceable transformation: raw text → 10 normalized tokens → 7 nouns (patient, john, smith, age, bp, chest, pain) → "140/90" extracted by regex. Every intermediate result is inspectable.

## spaCy Pipeline

The spaCy pipeline is like a food processor — one button does everything. `doc = nlp(text)` runs tokenization, POS tagging, dependency parsing, lemmatization, and NER in a single optimized pass. Results are all accessible as attributes on the same `Doc`/`Token` objects. The cost is less granular control (though you can disable or customize components with `nlp.disable()`).

An equivalent spaCy pipeline function is noticeably shorter. One `nlp(text)` call replaces multiple explicit steps. Nouns come from filtering `token.pos_ == "NOUN"` on the already-annotated Doc — no separate tokenize-then-tag step. Entities come from iterating `doc.ents` — no tree-walking. The tradeoff: if you need to customize stopword handling or swap the lemmatizer, you're working against the integrated pipeline rather than with it.

Regex extraction (`re.findall()`) is still a manual addition — spaCy doesn't automatically extract vitals or dosages by pattern. That's where `re.findall()` or spaCy's `Matcher` fills the gap.

## Comparing the Approaches

The key differences between the two pipeline approaches:

- **Assembly**: NLTK requires explicit chaining of each step (tokenize → filter → lemmatize → tag → chunk). spaCy runs everything in one `nlp(text)` call.
- **Customization**: NLTK lets you swap any component (different tokenizer, different stemmer). spaCy uses `nlp.disable()` to turn pipeline stages on or off, but individual components are harder to replace.
- **Stopwords**: NLTK requires building a set and filtering manually. spaCy tags each token with `token.is_stop` — a boolean you can filter on.
- **NER quality**: NLTK uses an older MaxEnt chunker; spaCy uses a neural model trained on OntoNotes. spaCy's entity recognition is substantially better on well-formed text.
- **Speed**: spaCy is faster, especially at scale — its pipeline is implemented in Cython and processes text in a single pass.

Use NLTK when you're exploring and need to understand what each step does (which is why we start with it in class). Use spaCy when you need to process text at scale and want reliable, fast, production-quality results. In practice, many projects use both — NLTK for prototyping and exploration, spaCy for the production system.

Both pipelines leave regex as a manual addition. Neither NLTK nor spaCy automatically extracts vitals, dosages, or other format-based patterns. That's where `re.findall()` complements the NLP pipeline.

# LIVE DEMO!!!

Demo 3 (Comparing Literary Worlds) is the capstone. Four literary texts (Alice in Wonderland, Sherlock Holmes, Pride and Prejudice, Frankenstein) become a mini corpus. Students vectorize with BoW and TF-IDF, examine which terms are most distinctive per text (expect "Watson" for Holmes, "creature" for Frankenstein, "Darcy" for P&P), compute pairwise cosine similarity, and visualize the similarity matrix as a heatmap. Then they build both NLTK and spaCy pipeline functions on the same corpus to compare the approaches end-to-end.

The mini-challenge: add a fifth text and see where it clusters. Another detective novel should land near Holmes; another gothic novel near Frankenstein. If it doesn't, that raises a productive question — TF-IDF similarity is driven by vocabulary overlap, not genre understanding. Two detective novels might share plot structure but use entirely different vocabularies, making them no more "similar" than any random pair by this measure. That's a concrete illustration of what BoW/TF-IDF can and can't capture.
