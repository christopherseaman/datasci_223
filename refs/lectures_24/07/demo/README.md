# API Prompt Engineering Demo

This directory contains materials for the API Prompt Engineering demo, focusing on using language model APIs for healthcare applications.

## Files

- `03-api_prompt_engineering.md` - The main demo markdown file
- `check_api_calls.py` - Script to check API calls in the markdown file
- `check_json_format.py` - Script to test JSON formatting with the OpenAI API

## Testing the API Calls

Before running the full demo, you can use the provided scripts to check if the API calls are working correctly.

### Prerequisites

1. Make sure you have an OpenAI API key
2. Create a `.env` file in this directory with your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install required packages:
   ```
   pip install openai python-dotenv
   ```

### Option 1: Check JSON Formatting

To quickly check if the OpenAI API returns valid JSON responses:

```bash
python check_json_format.py
```

This script runs three simple tests to verify that the API returns properly formatted JSON responses.

### Option 2: Check All API Calls

To check all API calls in the markdown file:

```bash
python check_api_calls.py 03-api_prompt_engineering.md
```

This script extracts all Python code blocks from the markdown file and tests any that contain API calls.

### Option 3: Convert to Notebook and Execute

To convert the markdown file to a Jupyter notebook and execute it:

```bash
jupytext --to notebook 03-api_prompt_engineering.md --execute
```

This will create and execute `03-api_prompt_engineering.ipynb`.

## Troubleshooting

If you encounter JSON parsing errors:

1. Make sure you're using the latest version of the OpenAI Python library
2. Check that the `response_format={"type": "json_object"}` parameter is included in API calls that expect JSON responses
3. Verify your API key has the necessary permissions
4. Check for rate limiting issues if you're making many API calls

## Notes

- The demo uses `gpt-4o-mini` as the model, which is a smaller, more cost-effective model
- API calls are configured to use a temperature of 0.3 for more consistent responses
- The scripts handle errors gracefully and provide informative error messages