# Lecture 07 Demos

Three hands-on demos covering GPT internals, embeddings, and LLM API prompt engineering.

## Files

| Demo | File | Topic |
|:---|:---|:---|
| 1 | `01_microgpt.md` | Build and train a GPT from scratch (microGPT): character-level tokenizer, autograd, multi-head attention, name generation |
| 2 | `02_embeddings_finetuning.md` | Sentence embeddings, cosine similarity, semantic search; GPT-2 fine-tuning and hallucination demo |
| 3 | `03_api_prompt_engineering.md` | Zero/one/few-shot prompting, schema-based JSON extraction, chain-of-thought on a PMC-Patients case report |

## Setup

### Demo 1: microGPT

```bash
pip install matplotlib seaborn numpy
```

No GPU required. The demo downloads a names dataset (~32KB) and trains a small GPT for ~1000 steps on CPU (2-5 minutes). Trained weights are saved to `microgpt_model.json` (~50KB) — set `REBUILD = False` to skip retraining.

### Demo 2: Embeddings & Fine-Tuning

```bash
pip install sentence-transformers chromadb transformers datasets torch accelerate matplotlib seaborn numpy pandas
```

No API key required — embedding models and GPT-2 run locally. First run downloads model weights (~500MB for sentence-transformers, ~500MB for GPT-2). GPU (CUDA) and Apple Silicon (MPS) are auto-detected for faster fine-tuning.

### Demo 3: API Prompt Engineering

```bash
pip install openai python-dotenv
```

**API key setup** — we use [OpenRouter](https://openrouter.ai) (OpenAI-compatible API, many models):

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Create an API key under "Keys"
3. Create a `.env` file in this directory:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

The demo uses `openai/gpt-4o-mini` via OpenRouter. OpenAI API keys also work as a fallback (set `OPENAI_API_KEY` instead).

## Converting and Running

```bash
# Convert all demos to notebooks
jupytext --to notebook 01_microgpt.md 02_embeddings_finetuning.md 03_api_prompt_engineering.md

# Or convert and execute in one step
jupytext --to notebook --execute 01_microgpt.md
```
