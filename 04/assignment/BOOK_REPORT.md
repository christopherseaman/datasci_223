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

# The Lazy Book Report

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
# Your code here: compute text statistics
# You'll need: import string, import re
# - Tokenize: remove punctuation, lowercase
# - Sentences: split on sentence-ending punctuation
# Calculate vocab_richness, avg_sentence_length, avg_word_length


```

---

## Question 2: Main Characters

> "Who are the main characters in this story?"

**NLP Method:** Use Named Entity Recognition (NER) to extract PERSON entities.

**Hint:** Use spaCy's `en_core_web_sm` model. Process the text and filter entities where `ent.label_ == 'PERSON'`. Count how often each name appears.

```python
# Your code here: extract PERSON entities using spaCy NER
# You'll need: import spacy, nlp = spacy.load("en_core_web_sm")

# When done, save your findings:
# with open("output/characters.txt", "w") as f:
#     for name in your_character_list:
#         f.write(f"{name}\n")


```

---

## Question 3: Story Locations

> "Where does the story take place?"

**NLP Method:** Use Named Entity Recognition (NER) to extract location entities (GPE and LOC).

**Hint:** Filter entities where `ent.label_` is 'GPE' (geopolitical entity) or 'LOC' (location).

```python
# Your code here: extract GPE and LOC entities using spaCy NER

# When done, save your findings:
# with open("output/locations.txt", "w") as f:
#     for place in your_locations_list:
#         f.write(f"{place}\n")


```

---

## Question 4: Wilson's Business

> "What is Wilson's business?"

**NLP Method:** Use TF-IDF similarity to find which section discusses Wilson's business.

**Hint:** Create a TF-IDF vectorizer, fit it on the 3 sections, then transform your query using the same vectorizer (`.transform()`, not `.fit_transform()` - you want to use the vocabulary learned from the sections). Find which section has the highest cosine similarity and read it to find the answer.

```python
# Your code here: use TF-IDF similarity to find the relevant section
# You'll need: from sklearn.feature_extraction.text import TfidfVectorizer
#              from sklearn.metrics.pairwise import cosine_similarity

# When done, save your findings:
# with open("output/business.txt", "w") as f:
#     f.write("Wilson's business is: ...")


```

---

## Question 5: Wilson's Work Routine

> "What is Wilson's daily work routine for the League?"

**NLP Method:** Use TF-IDF similarity to find which section discusses Wilson's work routine.

**Hint:** Similar to Question 4 - use TF-IDF to find the section that best matches your query about work routine. The answer includes what Wilson had to do and what eventually happened.

```python
# Your code here: use TF-IDF similarity to find the relevant section

# When done, save your findings:
# with open("output/routine.txt", "w") as f:
#     f.write("Wilson's work routine: ...\n")
#     f.write("What happened: ...\n")


```
