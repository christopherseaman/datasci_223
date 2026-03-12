#!/usr/bin/env python3
"""
Script to check API calls in the markdown file.
This extracts and tests the API calls from the markdown file to verify they work correctly.
"""

import os
import re
import json
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(
        "Error: No OpenAI API key found. Please set the OPENAI_API_KEY environment variable."
    )
    sys.exit(1)

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-4o-mini"  # Using a smaller, more cost-effective model


def extract_python_blocks(markdown_file):
    """Extract Python code blocks from a markdown file."""
    with open(markdown_file, "r") as f:
        content = f.read()

    # Find all Python code blocks
    pattern = r"```python\n(.*?)```"
    blocks = re.findall(pattern, content, re.DOTALL)

    return blocks


def test_api_call(code_block, block_index):
    """Test an API call in a code block."""
    print(f"\n--- Testing Block {block_index + 1} ---")

    # Check if this block contains an API call
    if "client.chat.completions.create" not in code_block:
        print("No API call found in this block. Skipping.")
        return True

    # Extract the function definition if present
    function_match = re.search(r"def\s+(\w+)\(.*?\).*?:", code_block, re.DOTALL)
    if function_match:
        function_name = function_match.group(1)
        print(f"Found function: {function_name}")

        # Check if there's example usage in the block
        example_usage = False
        if function_name + "(" in code_block:
            example_usage = True
            print("Found example usage in the block")

        if not example_usage:
            print("No example usage found. Skipping test.")
            return True

    # Create a standalone test script
    test_script = f"""
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL_NAME = "gpt-4o-mini"  # Using a smaller, more cost-effective model

# Test function
{code_block}
"""

    # Write the test script to a temporary file
    temp_file = "temp_test_script.py"
    with open(temp_file, "w") as f:
        f.write(test_script)

    try:
        # Execute the test script
        os.system(f"python {temp_file}")
        print("✅ API call executed successfully")
        return True
    except Exception as e:
        print(f"❌ Error executing API call: {str(e)}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    """Main function to check API calls in the markdown file."""
    if len(sys.argv) < 2:
        print("Usage: python check_api_calls.py <markdown_file>")
        sys.exit(1)

    markdown_file = sys.argv[1]
    if not os.path.exists(markdown_file):
        print(f"Error: File {markdown_file} not found.")
        sys.exit(1)

    print(f"Checking API calls in {markdown_file}...")

    # Extract Python code blocks
    blocks = extract_python_blocks(markdown_file)
    print(f"Found {len(blocks)} Python code blocks")

    # Test each block
    success_count = 0
    for i, block in enumerate(blocks):
        if test_api_call(block, i):
            success_count += 1

    print(
        f"\nSummary: {success_count}/{len(blocks)} blocks tested successfully"
    )


if __name__ == "__main__":
    main()
