# Assignment 7 Hints

## Part 1: Entity Extraction

### `build_prompt`
- Use an f-string with the note text embedded
- Include the exact JSON schema you want back (diagnosis, medications, lab_values, confidence)
- For few-shot mode, add 1-2 complete input/output examples before the target note
- Be explicit about the output format — LLMs follow instructions better when you show the schema

### `parse_json_response`
- LLMs often wrap JSON in markdown code blocks: `` ```json ... ``` ``
- Try `json.loads()` first on the raw text
- If that fails, look for content between `{` and `}` (find first `{`, find last `}`, slice)
- You can also strip markdown fences with string methods
- Return `None` on any parsing failure

### `validate_response`
- Check that the input is a dict
- Check for all four required keys: `diagnosis`, `medications`, `lab_values`, `confidence`
- You can use `all(key in response for key in required_keys)`

### `extract_entities`
- This is the orchestrator — call the other functions in sequence
- Handle the case where `parse_json_response` returns `None`

## Part 2: Semantic Search

### `embed_notes`
- `SentenceTransformer("all-MiniLM-L6-v2")` — first run downloads the model (~80MB)
- `model.encode(notes)` returns a numpy array directly

### `find_similar`
- You need to embed the query with the same model: `model.encode([query])`
- `cosine_similarity(query_embedding, embeddings)` returns a 2D array — take `[0]` for the 1D scores
- Use `sorted()` or `np.argsort()` to rank by score

## Part 3: Tiny LLM (optional)

Part 3 provides full working code — just read through it, run each cell, and experiment. The code is heavily commented to explain what each piece does.

- Both datasets (`dnd_spells.lst` and `icecream_flavors.lst`) work the same way — just change the filename
- If loss stays above ~3.0 after 1000 steps, make sure you ran the model definition cell first
- The ice cream dataset is smaller (450 items vs 518), so it may train faster but produce less variety
- `batch_first=True` in `TransformerDecoderLayer` means input shape is `(batch, sequence, embedding)` — this is already set in the provided code

## Common Issues

- **`ModuleNotFoundError: sentence_transformers`**: Run `pip install sentence-transformers`
- **First embedding is slow**: The model downloads on first use (~80MB). Subsequent runs use the cache.
- **API key not found**: Make sure `OPENROUTER_API_KEY` is set in your environment or `.env` file
- **JSON parsing fails**: Print the raw LLM response to see what format it's returning
