---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Assignment 7: Clinical NLP with LLMs and Embeddings

Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

**Dataset:** 75 synthetic discharge summaries from [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/aisc-team-a1/Asclepius-Synthetic-Clinical-Notes) (Kweon et al., 2023) in `asclepius_notes.json`.

## Setup

```python
%pip install -q -r requirements.txt

# Clear state after installing packages. If you re-run cells out of order later, re-run this cell first.
%reset -f
```

```python
import os
import json
import random
import numpy as np
from dotenv import load_dotenv

os.makedirs("output", exist_ok=True)
load_dotenv()
print("Setup complete!")
```

### API Key

Part 1 requires an [OpenRouter](https://openrouter.ai) API key (OpenAI keys also work). Copy the example and fill in your key:

```bash
cp example.env .env
# Then edit .env with your actual key
```

Part 2 runs locally and does not need an API key.

### Helper Functions (do not modify)

```python
# --- LLM client setup (do not modify) ---

def get_client():
    """Initialize the LLM client based on available API keys."""
    from openai import OpenAI

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        return client, "openrouter"

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "openai"

    raise ValueError(
        "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
    )


def call_llm(prompt, provider, client):
    """Send a prompt to the LLM and return the response text."""
    model = "openai/gpt-4o-mini" if provider == "openrouter" else "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a medical information extraction assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content


def get_device():
    """Detect the best available device for local model inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
```

### Load Data

```python
with open("asclepius_notes.json") as f:
    asclepius = json.load(f)

print(f"Loaded {len(asclepius)} synthetic clinical notes")
print(f"Keys: {list(asclepius[0].keys())}")
```

```python
print(asclepius[0]["note"][:500] + "...")
```

---

## Part 1: Clinical Entity Extraction

Use LLM prompt engineering to extract structured medical data from clinical notes.

```python
# Select 4 notes for extraction
random.seed(2026)
sample = random.sample(asclepius, 4)
notes_p1 = [s["note"] for s in sample]

print(f"Selected {len(notes_p1)} notes for extraction")
for i, n in enumerate(notes_p1, 1):
    print(f"\n--- Note {i} ({len(n)} chars) ---")
    print(n[:150] + "...")
```

### `build_prompt`

Build a prompt that instructs the LLM to extract structured data from a clinical note.

```python
# TODO: Implement build_prompt
# Requirements:
#   - Describe the extraction task clearly
#   - Specify the JSON output schema with these fields:
#     {"diagnosis": str, "medications": list, "lab_values": dict, "confidence": float}
#   - When few_shot=True, include 1-2 example input/output pairs
#   - Include the clinical note text
def build_prompt(note, few_shot=False):
    pass  # replace with your implementation
```

### `parse_json_response`

Extract a JSON object from LLM response text, which may contain markdown code fences or other wrapping.

```python
# TODO: Implement parse_json_response
# Requirements:
#   - Handle clean JSON strings (direct json.loads)
#   - Handle JSON wrapped in ```json ... ``` markdown blocks
#   - Find JSON within surrounding text (look for outermost { and })
#   - Return None if parsing fails
def parse_json_response(text):
    pass  # replace with your implementation
```

### `validate_response`

Check that a parsed response dict contains all required keys.

```python
# TODO: Implement validate_response
# Required fields: diagnosis, medications, lab_values, confidence
# Return True if all present, False otherwise
def validate_response(response):
    pass  # replace with your implementation
```

### `extract_entities`

Orchestrate the full extraction pipeline: get client, build prompt, call LLM, parse, validate, return.

```python
# TODO: Implement extract_entities
# Steps:
#   1. client, provider = get_client()
#   2. prompt = build_prompt(note, few_shot=few_shot)
#   3. raw = call_llm(prompt, provider=provider, client=client)
#   4. parsed = parse_json_response(raw)
#   5. Validate and return (return None if parsing or validation fails)
def extract_entities(note, few_shot=False):
    pass  # replace with your implementation
```

### Test extraction

```python
results_p1 = []
for i, note in enumerate(notes_p1, 1):
    result = extract_entities(note, few_shot=True)
    print(f"--- Note {i} ---")
    if result:
        print(json.dumps(result, indent=2))
        results_p1.append(result)
    else:
        print("Extraction failed")
    print()
```

### Save Part 1 results (do not modify)

```python
with open("output/extraction_results.json", "w") as f:
    json.dump(results_p1, f, indent=2)

print(f"Saved {len(results_p1)} extraction results to output/extraction_results.json")
```

---

## Part 2: Semantic Search

Build a semantic search system that finds clinical notes by meaning rather than keywords, using sentence embeddings and cosine similarity.

This part runs locally — no API key needed.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
print(f"Model loaded on {get_device()}")
```

```python
# Use all 75 notes for the search corpus
notes_p2 = [n["note"] for n in asclepius]
print(f"{len(notes_p2)} notes in search corpus")
```

### `embed_notes`

Generate embeddings for a list of notes using the sentence transformer model.

```python
# TODO: Implement embed_notes
# Use model.encode(notes) — returns a numpy array of shape (n_notes, embedding_dim)
def embed_notes(notes):
    pass  # replace with your implementation
```

### `find_similar`

Search notes by meaning using cosine similarity.

```python
# TODO: Implement find_similar
# Steps:
#   1. Embed the query with model.encode([query])
#   2. Compute cosine_similarity(query_embedding, embeddings)
#   3. Sort by score descending
#   4. Return top_k results as [{"note": str, "score": float}, ...]
#
# Note: this function uses the `model` variable from notebook scope.
# This is a common notebook pattern — the model is loaded once and reused
# across cells. Outside a notebook you'd pass the model as a parameter.
def find_similar(query, notes, embeddings, top_k=2):
    pass  # replace with your implementation
```

### Run the search pipeline

```python
embeddings = embed_notes(notes_p2)
print(f"Embeddings: {embeddings.shape}")

queries = [
    "heart attack symptoms",
    "infectious disease with fever",
    "respiratory illness",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    results = find_similar(q, notes_p2, embeddings, top_k=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. (score: {r['score']:.3f}) {r['note'][:80]}...")
```

### Save Part 2 results (do not modify)

```python
search_results = find_similar("heart attack symptoms", notes_p2, embeddings, top_k=3)
with open("output/search_results.json", "w") as f:
    json.dump(search_results, f, indent=2)

print(f"Saved {len(search_results)} search results to output/search_results.json")
```

---

## Validation

```python
print("Run 'python -m pytest .github/tests/ -v' in your terminal to check your work.")
```

---

## Part 3: Build a Tiny LLM *(optional, not graded)*

Train a character-level transformer to generate new text from a dataset of short strings. This mirrors the microGPT demo from lecture — same architecture, different data, using PyTorch's built-in modules instead of writing everything from scratch.

**Choose your dataset** (or use both!):

| Dataset | File | Items | Description |
|:---|:---|:---|:---|
| D&D Spells | `dnd_spells.lst` | 518 | Official spell names from Dungeons & Dragons |
| Ice Cream | `icecream_flavors.lst` | 450 | Ice cream flavor names from a [CMU student survey](https://www.cs.cmu.edu/~15110-f23/slides/all-icecream.csv) |

The code below uses D&D spells — swap the filename and variable names if you prefer ice cream.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
```

### Load and prepare data

```python
# Choose your dataset: "dnd_spells.lst" or "icecream_flavors.lst"
datafile = "dnd_spells.lst"

with open(datafile) as f:
    lines = f.read().strip().split("\n")
items = [line.strip() for line in lines[1:] if line.strip()]  # skip header

text = "\n".join(items)
chars = sorted(set(text))
vocab_size = len(chars)

# Character <-> integer mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
print(f"{len(items)} items from {datafile}")
print(f"{len(chars)} unique characters, {len(data)} total tokens")
print(f"Vocabulary: {''.join(chars)}")
```

### Define the model

This is a minimal GPT: token embeddings + position embeddings → transformer decoder → output head. Read through the code, then run the cell.

```python
block_size = 32   # context window (characters)
n_embd = 64       # embedding dimension
n_head = 4        # attention heads
n_layer = 2       # transformer blocks
dropout = 0.1


class CharGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Each character gets a learnable vector of size n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # Each position (0..block_size-1) also gets a learnable vector
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Stack of transformer decoder layers — this is where attention happens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        self.ln = nn.LayerNorm(n_embd)
        # Project from embedding space back to vocabulary size (one logit per character)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)                                    # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))    # (T, n_embd)
        x = self.drop(tok + pos)                                   # (B, T, n_embd)

        # Causal mask: prevents each position from attending to future positions
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        x = self.ln(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


device = get_device()
char_model = CharGPT().to(device)
print(f"CharGPT: {sum(p.numel() for p in char_model.parameters()):,} parameters on {device}")
```

### Train

The training loop samples random chunks from the data and teaches the model to predict the next character. Loss should drop below ~2.0 after 2000 steps.

```python
optimizer = torch.optim.AdamW(char_model.parameters(), lr=3e-4)
batch_size = 32
steps = 2000

for step in range(steps):
    # Pick random starting positions
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

    logits, loss = char_model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0 or step == steps - 1:
        print(f"step {step:4d} | loss {loss.item():.4f}")
```

### Generate

Sample from the trained model at different temperatures. Lower temperature = more conservative (common patterns), higher = more creative (weirder output).

```python
@torch.no_grad()
def generate(model, max_new_tokens=500, temperature=0.8):
    model.eval()
    idx = torch.tensor([[stoi["\n"]]], device=device)
    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return decode(idx[0].tolist())


for temp in [0.5, 0.8, 1.2]:
    print(f"\n--- Temperature {temp} ---")
    output = generate(char_model, temperature=temp)
    names = [s.strip() for s in output.split("\n") if s.strip()]
    for name in names[:10]:
        print(f"  {name}")
```

### Experiment (optional)

Try changing things and see what happens:

- Switch datasets — do ice cream flavors vs spell names produce different quality output?
- Increase `n_layer` to 4 or `n_embd` to 128 — does the model improve? How much slower is training?
- Train for 5000 steps instead of 2000
- What happens at very low temperature (0.2) vs very high (2.0)?
