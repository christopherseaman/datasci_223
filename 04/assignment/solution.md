---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# The Lazy Book Report - SOLUTION

Your professor has assigned a book report on "The Red-Headed League" by Arthur Conan Doyle. 

You haven't read the book. And out of stubbornness, you won't.

But you *have* learned NLP. Let's use it to answer the professor's questions without reading.

## Setup

First, let's fetch the text from Project Gutenberg and prepare it for analysis.

```python
# Fetch and prepare text - RUN THIS CELL FIRST
import os
import urllib.request
import re

os.makedirs("output", exist_ok=True)

url = 'https://www.gutenberg.org/files/1661/1661-0.txt'
req = urllib.request.Request(url, headers={'User-Agent': 'Python-urllib'})
with urllib.request.urlopen(req, timeout=30) as resp:
    text = resp.read().decode('utf-8')

# Strip Gutenberg boilerplate
text = text.split('*** START OF')[1].split('***')[1]
text = text.split('*** END OF')[0]

# Extract "The Red-Headed League" story (it's the second story in the collection)
matches = list(re.finditer(r'THE RED-HEADED LEAGUE', text, re.IGNORECASE))
story_start = matches[1].end()
story_text = text[story_start:]
story_end = re.search(r'\n\s*III\.\s*\n', story_text)
story_text = story_text[:story_end.start()] if story_end else story_text

# Split into 3 sections by word count
words = story_text.split()[:4000]
section_size = len(words) // 3
sections = [
    ' '.join(words[:section_size]),
    ' '.join(words[section_size:2*section_size]),
    ' '.join(words[2*section_size:])
]

print(f"Story loaded: {len(words)} words in {len(sections)} sections")
print(f"Section sizes: {[len(s.split()) for s in sections]}")
```

## Professor's Questions

Your professor wants you to answer 5 questions about the story. Let's use NLP to find the answers.

---

## Question 1: Writing Style

> "This text is from the 1890s. What makes it different from modern writing?"

**NLP Method:** Use preprocessing to compute text statistics. Tokenize the text and calculate:
- Vocabulary richness (unique words / total words)
- Average sentence length
- Average word length

**Hint:** Formal, literary writing typically shows higher vocabulary richness and longer sentences than modern casual text.

```python
# SOLUTION: compute text statistics
import string

# Tokenize and normalize
punct = string.punctuation
tokens = [w.lower().strip(punct) for w in words if w.strip(punct)]

# Calculate vocabulary richness
unique_words = len(set(tokens))
total_words = len(tokens)
vocab_richness = unique_words / total_words

# Calculate average word length
avg_word_length = sum(len(w) for w in tokens) / len(tokens)

# Calculate average sentence length
full_text = ' '.join(words)
sentences = re.split(r'[.!?]+\s+', full_text)
sentences = [s for s in sentences if s.strip()]
avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

print(f"Vocabulary richness: {vocab_richness:.1%}")
print(f"Average sentence length: {avg_sentence_length:.1f} words")
print(f"Average word length: {avg_word_length:.2f} characters")
print(f"\nInterpretation: High vocabulary richness and long sentences")
print(f"indicate formal, literary 19th-century writing style.")
```

---

## Question 2: Main Characters

> "Who are the main characters in this story?"

**NLP Method:** Use Named Entity Recognition (NER) to extract PERSON entities.

**Hint:** Use spaCy's `en_core_web_sm` model. Process the text and filter entities where `ent.label_ == 'PERSON'`. Count how often each name appears.

```python
# SOLUTION: extract PERSON entities using spaCy NER
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Process full text (limit to avoid memory issues)
doc = nlp(full_text[:50000])

# Extract PERSON entities
persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
person_counts = Counter(persons)

print("Main characters found (by mention count):")
main_characters = [name for name, count in person_counts.most_common(10) if count >= 2]
for name in main_characters:
    print(f"  {name}")

# Save to file
with open("output/characters.txt", "w") as f:
    for name in main_characters:
        f.write(f"{name}\n")
print("\nSaved to output/characters.txt")
```

---

## Question 3: Story Locations

> "Where does the story take place?"

**NLP Method:** Use Named Entity Recognition (NER) to extract location entities (GPE and LOC).

**Hint:** Filter entities where `ent.label_` is 'GPE' (geopolitical entity) or 'LOC' (location).

```python
# SOLUTION: extract GPE and LOC entities using spaCy NER

# Extract location entities
locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
location_counts = Counter(locations)

print("Locations found:")
main_locations = [place for place, count in location_counts.most_common(10)]
for place in main_locations:
    print(f"  {place}")

# Save to file
with open("output/locations.txt", "w") as f:
    for place in main_locations:
        f.write(f"{place}\n")
print("\nSaved to output/locations.txt")
```

---

## Question 4: Wilson's Business

> "What is Wilson's business?"

**NLP Method:** Use TF-IDF similarity to find which section discusses Wilson's business.

**Hint:** Create a TF-IDF vectorizer, fit it on the 3 sections, then transform your query using the same vectorizer (`.transform()`, not `.fit_transform()` - you want to use the vocabulary learned from the sections). Find which section has the highest cosine similarity and read it to find the answer.

```python
# SOLUTION: use TF-IDF similarity to find the relevant section
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create TF-IDF vectors for sections
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = vectorizer.fit_transform(sections)

# Query for Wilson's business
query = "What is Wilson's business?"
query_vec = vectorizer.transform([query])
similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

print(f"Query: '{query}'")
print(f"Similarities: Section 1={similarities[0]:.3f}, Section 2={similarities[1]:.3f}, Section 3={similarities[2]:.3f}")

best_section = similarities.argmax() + 1
print(f"Best match: Section {best_section}")

# Find the answer in the best section
best_text = sections[best_section - 1]
if 'pawnbroker' in best_text.lower():
    idx = best_text.lower().find('pawnbroker')
    snippet = best_text[max(0, idx-20):idx+60]
    print(f"\nFound in text: '...{snippet}...'")

# Save to file
with open("output/business.txt", "w") as f:
    f.write("Wilson is a pawnbroker.\n")
print("\nSaved to output/business.txt")
```

---

## Question 5: Wilson's Work Routine

> "What is Wilson's daily work routine for the League?"

**NLP Method:** Use TF-IDF similarity to find which section discusses Wilson's work routine.

**Hint:** Similar to Question 4 - use TF-IDF to find the section that best matches your query about work routine. The answer includes what Wilson had to do and what eventually happened.

```python
# SOLUTION: use TF-IDF similarity to find the relevant section

# Query for work routine
query5 = "What is Wilson's daily work routine for the League?"
query_vec5 = vectorizer.transform([query5])
similarities5 = cosine_similarity(query_vec5, tfidf_matrix)[0]

print(f"Query: '{query5}'")
print(f"Similarities: Section 1={similarities5[0]:.3f}, Section 2={similarities5[1]:.3f}, Section 3={similarities5[2]:.3f}")

best_section5 = similarities5.argmax() + 1
print(f"Best match: Section {best_section5}")

# Find details in the best section
best_text5 = sections[best_section5 - 1]

if 'copy' in best_text5.lower():
    idx = best_text5.lower().find('copy')
    snippet = best_text5[max(0, idx-10):idx+80]
    print(f"\nWork routine: '...{snippet}...'")

if 'dissolved' in best_text5.lower():
    idx = best_text5.lower().find('dissolved')
    snippet = best_text5[max(0, idx-30):idx+50]
    print(f"\nWhat happened: '...{snippet}...'")

# Save to file
with open("output/routine.txt", "w") as f:
    f.write("Wilson's work routine: copying the Encyclopaedia Britannica\n")
    f.write("What happened: The Red-Headed League was dissolved\n")
print("\nSaved to output/routine.txt")
```
