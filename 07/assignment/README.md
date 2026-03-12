# Assignment 7: Clinical NLP with LLMs and Embeddings

Extract structured data from clinical notes using LLM prompt engineering, then build a semantic search system using sentence embeddings.

**Dataset**: 75 synthetic discharge summaries from [Asclepius-Synthetic-Clinical-Notes](https://huggingface.co/datasets/aisc-team-a1/Asclepius-Synthetic-Clinical-Notes) (Kweon et al., 2023) in `asclepius_notes.json`.

## Getting Started

```bash
pip install -r requirements.txt
```

### API Key (Part 1 only)

Part 1 requires an LLM API key. We will use [OpenRouter](https://openrouter.ai) (OpenAI-compatible):

1. An API key will be provided on the class forum.
2. (Optional) Sign up for your own free OpenRouter account to create your own API key. There are usually generous free-tier limits for a few models at any given time.
3. Save the API key in `.env` as `OPENROUTER_API_KEY`. For example:

   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

4. DO NOT COMMIT YOUR API KEY. This will likely invalidate the shared key for everyone and I will have to generate a new one. **This will result in an immediate deduction of one million imaginary points from your final grade**.

Part 2 (embeddings) runs locally and does not need an API key.

## Workflow

Open `assignment.ipynb` and work through both parts:

1. **Part 1: Clinical Entity Extraction** — Implement `build_prompt`, `parse_json_response`, `validate_response`, and `extract_entities`
2. **Part 2: Semantic Search** — Implement `embed_notes` and `find_similar`
3. **Part 3: Build a Tiny LLM** *(optional, not graded)* — Train a character-level transformer to generate D&D spell names or ice cream flavors

The notebook saves results to `output/` for autograding. Note that there are several helper functions provided for "boilerplate" tasks like creating the LLM client, making the API call, and checking for acceleration support. You should definitely play with these and understand how they work, but it is a best practice to abstract away these details so you can focus on the core logic of your solution.

## Output Files

| File | Part | Description |
|:---|:---|:---|
| `output/extraction_results.json` | 1 | Structured entities extracted from clinical notes |
| `output/search_results.json` | 2 | Semantic search results with similarity scores |

## Checking Your Work

```bash
python -m pytest .github/tests/ -v
```

Note: Tests check output artifacts only — run the notebook first, then run tests. You must commit `output/` for CI autograding to pass. Part 3 is optional and has no tests.

## Hints

See `hints.md` for troubleshooting tips.
