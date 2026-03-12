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

# Assignment 7: Clinical NLP with LLMs and Embeddings — Solution

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

```python
random.seed(2026)
sample = random.sample(asclepius, 4)
notes_p1 = [s["note"] for s in sample]

print(f"Selected {len(notes_p1)} notes for extraction")
for i, n in enumerate(notes_p1, 1):
    print(f"\n--- Note {i} ({len(n)} chars) ---")
    print(n[:150] + "...")
```

### `build_prompt`

```python
def build_prompt(note, few_shot=False):
    if few_shot:
        return f"""Extract structured medical data from clinical notes. Return JSON only.

Example:
Note: "72-year-old male with progressive dyspnea and orthopnea. BNP 1840 pg/mL, ejection fraction 25%.
Chest X-ray shows cardiomegaly with pulmonary edema. Started on furosemide 40mg IV, lisinopril 5mg daily,
and carvedilol 6.25mg BID. Assessment: Acute decompensated heart failure."

Output:
{{
  "diagnosis": "Acute decompensated heart failure",
  "medications": ["furosemide 40mg IV", "lisinopril 5mg daily", "carvedilol 6.25mg BID"],
  "lab_values": {{"BNP": "1840 pg/mL", "ejection_fraction": "25%"}},
  "confidence": 0.95
}}

Now extract from this note:
Note: "{note}"

Output:"""

    return f"""Extract structured medical data from the following clinical note.
Return ONLY a JSON object with exactly these fields:

{{
  "diagnosis": "primary diagnosis as a string",
  "medications": ["list of medications with doses"],
  "lab_values": {{"test_name": "value with units"}},
  "confidence": 0.0
}}

Rules:
- confidence is a float from 0.0 to 1.0
- include ALL medications mentioned with doses if given
- include ALL lab values with units
- if a field has no data, use an empty list [] or empty dict {{}}

Clinical note:
{note}"""
```

### `parse_json_response`

```python
def parse_json_response(text):
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end]
            lines = block.split("\n")
            block = "\n".join(lines[1:])
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None
```

### `validate_response`

```python
def validate_response(response):
    if not isinstance(response, dict):
        return False
    required = {"diagnosis", "medications", "lab_values", "confidence"}
    return required.issubset(response.keys())
```

### `extract_entities`

```python
def extract_entities(note, few_shot=False):
    client, provider = get_client()
    prompt = build_prompt(note, few_shot=few_shot)
    raw = call_llm(prompt, provider=provider, client=client)
    parsed = parse_json_response(raw)
    if parsed is None:
        return None
    if not validate_response(parsed):
        return None
    return parsed
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

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2", device=get_device())
print(f"Model loaded on {get_device()}")
```

```python
notes_p2 = [n["note"] for n in asclepius]
print(f"{len(notes_p2)} notes in search corpus")
```

### `embed_notes`

```python
def embed_notes(notes):
    return model.encode(notes)
```

### `find_similar`

```python
def find_similar(query, notes, embeddings, top_k=2):
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    ranked = sorted(
        zip(notes, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"note": n, "score": float(s)} for n, s in ranked[:top_k]]
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
items = [line.strip() for line in lines[1:] if line.strip()]

text = "\n".join(items)
chars = sorted(set(text))
vocab_size = len(chars)

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

```python
block_size = 32
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1


class CharGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        x = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


device = get_device()
char_model = CharGPT().to(device)
print(f"CharGPT: {sum(p.numel() for p in char_model.parameters()):,} parameters on {device}")
```

### Train

```python
optimizer = torch.optim.AdamW(char_model.parameters(), lr=3e-4)
batch_size = 32
steps = 2000

for step in range(steps):
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
