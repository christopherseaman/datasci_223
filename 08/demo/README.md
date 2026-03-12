# Lecture 08 Demos: LLM Applications & Workflows

## Setup

```bash
pip install -r requirements.txt
```

### API Key

Demos 1-3 require an LLM API key. Copy the example and add your key:

```bash
cp example.env .env
# Edit .env with your OPENROUTER_API_KEY or OPENAI_API_KEY
```

### Convert to Notebooks

```bash
jupytext --to notebook *.md
```

## Demos

| Demo | File | Topics |
|:---|:---|:---|
| 1 | `01_agents.md` | Agents SDK, specialized agents (diagnostician, pharmacist, summarizer), structured output (`output_type`) |
| 2 | `02_rag_pipeline.md` | Clinical knowledge base, ChromaDB, RAG pipeline (embed/retrieve/augment/generate), citations, function calling (tool use + structured output), MCP server/client |
| 3 | `03_workflows.md` | Prompt chaining, PHI guardrails, deterministic steps, failure modes, combined pipeline |

## Requirements

- Python 3.11+
- API key (OpenRouter or OpenAI)
- Node.js/npx (optional, for MCP server demo in Demo 2)
