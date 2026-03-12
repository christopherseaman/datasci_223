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

# Demo 1: Building a GPT from Scratch

Andrej Karpathy's [microGPT](https://karpathy.github.io/2026/02/12/microgpt/) is a working GPT in ~200 lines of pure Python with zero dependencies. We'll walk through it piece by piece, then train it and generate text.

The [full source](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) fits in a single file. Everything below is that file, broken into cells with commentary.

## Dataset

microGPT trains on a list of names — each name is a "document." The model learns character-level patterns: which letters tend to follow which.

```python
import os
import math
import random
random.seed(42)

# Download the names dataset
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
print(f"first 10: {docs[:10]}")
```

## Tokenizer

The simplest possible tokenizer: each unique character gets an integer ID. Plus one special token (BOS = Beginning of Sequence) to mark where names start and end.

Production models use **BPE (Byte Pair Encoding)** instead — grouping common character sequences into subword tokens. But the principle is the same: text in, integers out.

```python
uchars = sorted(set(''.join(docs)))  # unique characters become token ids 0..n-1
BOS = len(uchars)  # special Beginning of Sequence token
vocab_size = len(uchars) + 1

print(f"vocab size: {vocab_size}")
print(f"characters: {''.join(uchars)}")
print(f"BOS token id: {BOS}")

# Tokenize a name
name = docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
print(f"\n'{name}' -> {tokens}")
print(f"decoded: {''.join(uchars[t] if t != BOS else '<BOS>' for t in tokens)}")
```

## Autograd Engine

The `Value` class tracks every arithmetic operation and can compute gradients automatically via backpropagation. This is the same thing PyTorch does — but on individual scalars, so we can see every step.

Each `Value` stores:
- `data`: the scalar result of the forward pass
- `grad`: the derivative of the loss with respect to this value (filled in during backward pass)
- `_children` and `_local_grads`: the computation graph for backprop

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
```

### Quick autograd demo

Verify with a simple function — compute f(x,y) = x*y + x^2 and check the analytic gradients.

```python
# f(x, y) = x*y + x^2
# df/dx = y + 2x, df/dy = x
x = Value(3.0)
y = Value(4.0)
z = x * y + x ** 2  # 3*4 + 9 = 21

z.backward()
print(f"z = {z.data}")           # 21.0
print(f"dz/dx = {x.grad}")      # y + 2x = 4 + 6 = 10.0
print(f"dz/dy = {y.grad}")      # x = 3.0
```

## Model Parameters

Here's the full architecture we're building — every box below maps to code in this notebook:

![microGPT Architecture](../media/microgpt-arch.png)

Initialize all the learned weights. These random numbers are what the model "knows" — training will adjust them to capture patterns in the data.

The architecture follows GPT-2 (with minor simplifications):
- **Token embeddings** (`wte`): one vector per vocabulary item
- **Position embeddings** (`wpe`): one vector per position in the context window
- **Attention weights** (`attn_wq/wk/wv/wo`): the Q, K, V projections and output projection from the lecture
- **MLP weights** (`mlp_fc1/fc2`): the feed-forward network (expand 4x, ReLU, contract)
- **Output head** (`lm_head`): project back to vocabulary size for next-token prediction

```python
n_layer = 1       # number of transformer layers (GPT-2 uses 12-48)
n_embd = 16       # embedding dimension (GPT-2 uses 768-1600)
block_size = 16   # max context length (GPT-2 uses 1024)
n_head = 4        # attention heads (GPT-2 uses 12-25)
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),      # token embeddings
    'wpe': matrix(block_size, n_embd),       # position embeddings
    'lm_head': matrix(vocab_size, n_embd),   # output projection
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # query projection
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # key projection
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # value projection
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # output projection
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # expand
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # contract

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
print(f"for comparison: GPT-2 has 124M params, GPT-3 has 175B")
```

## The GPT Forward Pass

> **Try it live:** The [microGPT visualizer](https://microgpt.boratto.ca) lets you step through this exact forward pass interactively — watch tokens flow through embeddings, attention, and the MLP in real time. Open it in a second tab while reading the code below.

This is the core algorithm. For each token, the model:

1. Looks up token + position embeddings
2. Runs through each transformer layer:
   - **Multi-head attention**: each token asks "which other tokens should I pay attention to?"
   - **MLP**: process what attention found
   - **Residual connections**: add the input back (the `x + residual` pattern from the lecture)
3. Projects to vocabulary-sized logits → softmax → next-token probabilities

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # 1. Embed: token + position
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 2a. Multi-head attention
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # residual connection

        # 2b. MLP (feed-forward)
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]  # residual connection

    # 3. Project to vocabulary
    logits = linear(x, state_dict['lm_head'])
    return logits

print("Model defined. Forward pass ready.")
```

## Training

The training loop is the simplest possible version of what every LLM does:

1. **Tokenize** a document (name)
2. **Forward** each token through the model, predicting the next one
3. **Compute loss** — cross-entropy: how far were the predictions from the actual next token?
4. **Backward** — compute gradients of loss with respect to every parameter
5. **Update** — Adam optimizer adjusts parameters to reduce loss
6. Repeat

This will take a few minutes on CPU. Watch the loss decrease — that's the model learning patterns.

Set `REBUILD = False` to skip training and load a previously saved model. The trained weights are saved as JSON (~50 KB) — small enough to include in version control, so you can share a pre-trained model with classmates or reload between sessions without waiting.

```python
# Adam optimizer buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v_buf = [0.0] * len(params)

import json

MODEL_FILE = "microgpt_model.json"
REBUILD = True  # Set to False to load a previously saved model (if it exists)

num_steps = 1000
loss_history = []

if not REBUILD and os.path.exists(MODEL_FILE):
    # Load saved model weights
    with open(MODEL_FILE) as f:
        saved = json.load(f)
    for name, mat in saved["state_dict"].items():
        for i, row in enumerate(mat):
            for j, val in enumerate(row):
                state_dict[name][i][j] = Value(val)
    params[:] = [p for mat in state_dict.values() for row in mat for p in row]
    loss_history = saved.get("loss_history", [])
    print(f"Loaded model from {MODEL_FILE} ({len(params)} params, {len(loss_history)} training steps)")
else:
    print(f"Training for {num_steps} steps...")

    for step in range(num_steps):
        # Tokenize one document
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward pass: predict each next token
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        # Backward pass: compute gradients
        loss.backward()

        # Adam optimizer update
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        loss_history.append(loss.data)
        if step % 100 == 0 or step == num_steps - 1:
            print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

    # Save model weights
    saved_state = {}
    for name, mat in state_dict.items():
        saved_state[name] = [[v.data for v in row] for row in mat]
    with open(MODEL_FILE, "w") as f:
        json.dump({"state_dict": saved_state, "loss_history": loss_history}, f)
    print(f"Training complete! Model saved to {MODEL_FILE}")
    print(f"Model file size: {os.path.getsize(MODEL_FILE) / 1024:.1f} KB")
```

### Training Loss Curve

Plotting loss over training steps shows the model learning. The steep early drop is the model picking up basic character frequencies; the plateau is where it's learning subtler patterns like common letter combinations.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(loss_history, alpha=0.3, color='steelblue', label='Per-step loss')

# Smoothed curve (moving average)
window = 50
if len(loss_history) > window:
    smoothed = [sum(loss_history[i:i+window]) / window for i in range(len(loss_history) - window)]
    plt.plot(range(window, len(loss_history)), smoothed, color='darkblue', linewidth=2, label=f'{window}-step moving avg')

plt.xlabel('Training Step')
plt.ylabel('Cross-Entropy Loss')
plt.title('microGPT Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Inference: Generate New Names

Now the model "hallucinates" — it generates new names that don't exist in the training data but follow the patterns it learned. This is the same process as ChatGPT generating text, just at character-level scale.

**Temperature** controls randomness:
- Low (0.1) → conservative, repetitive
- High (1.0) → creative, chaotic

```python
def generate(temperature=0.5, num_samples=20):
    """Generate new names from the trained model."""
    samples = []
    for _ in range(num_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        chars = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            chars.append(uchars[token_id])
        samples.append(''.join(chars))
    return samples

print("--- Generated names (temperature=0.5) ---")
for i, name in enumerate(generate(0.5), 1):
    print(f"  {i:2d}. {name}")
```

```python
print("--- Temperature comparison ---\n")

for temp in [0.2, 0.5, 1.0]:
    names = generate(temp, num_samples=10)
    print(f"temperature={temp}:")
    for name in names:
        print(f"  {name}")
    print()
```

## Interactive Mode: Complete a Name

Type a starting prefix (or leave blank for a random start) and watch the model complete it character by character. This is the same autoregressive loop ChatGPT uses — just at character scale.

```python
def complete(prefix="", temperature=0.5, num_samples=5):
    """Generate name completions from a starting prefix."""
    samples = []
    for _ in range(num_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        # Feed prefix tokens
        if prefix:
            token_ids = [BOS] + [uchars.index(ch) for ch in prefix.lower() if ch in uchars]
        else:
            token_ids = [BOS]
        # Forward through prefix to build up KV cache
        for pos_id, token_id in enumerate(token_ids):
            logits = gpt(token_id, pos_id, keys, values)
        # Generate remaining characters
        chars = list(prefix.lower())
        token_id = random.choices(range(vocab_size), weights=[softmax([l / temperature for l in logits])[i].data for i in range(vocab_size)])[0]
        if token_id != BOS:
            chars.append(uchars[token_id])
            for pos_id in range(len(token_ids), block_size):
                logits = gpt(token_id, pos_id, keys, values)
                probs = softmax([l / temperature for l in logits])
                token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
                if token_id == BOS:
                    break
                chars.append(uchars[token_id])
        samples.append(''.join(chars))
    return samples

# Try it! Change the prefix to steer generation.
prefix = "ka"
print(f"Completions starting with '{prefix}':\n")
for i, name in enumerate(complete(prefix, temperature=0.5), 1):
    print(f"  {i}. {name}")
```

```python
# Edit these and re-run to explore! Try your own name, initials, or random letters.
for prefix in ["mar", "ch", "al", "zi"]:
    names = complete(prefix, temperature=0.5, num_samples=3)
    print(f"  '{prefix}' → {', '.join(names)}")
```

**Further reading:**
- [microGPT visualizer](https://microgpt.boratto.ca) — step through the internals interactively
- [microGPT blog post](https://karpathy.github.io/2026/02/12/microgpt/) — Karpathy's full writeup
- [nanoGPT repo](https://github.com/karpathy/nanoGPT) — same algorithm, optimized with PyTorch
