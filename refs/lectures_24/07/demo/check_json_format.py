#!/usr/bin/env python3
"""
Script to check if the OpenAI API returns valid JSON responses.
This script tests a few representative API calls to verify JSON formatting works.
"""

import os
import json
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
    exit(1)

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-4o-mini"  # Using a smaller, more cost-effective model


def test_json_response(prompt, system_role="You are a helpful assistant."):
    """Test if the API returns a valid JSON response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},  # Request JSON response
            temperature=0.3,
        )

        # Try to parse the JSON response
        content = response.choices[0].message.content
        json_data = json.loads(content)

        return True, json_data
    except Exception as e:
        return False, str(e)


def main():
    """Run tests to check JSON formatting."""
    print("Testing OpenAI API JSON responses...")

    # Test 1: Simple classification
    print("\nTest 1: Medical Text Classification")
    prompt = """Classify the following medical text into one of these categories: Diagnosis, Treatment, Prognosis, Medical History.
    
Text: Patient presents with persistent cough and fever for 3 days. Chest X-ray shows right lower lobe infiltrate. Started on azithromycin 500mg daily.

Provide the classification in JSON format with the following structure:
{
    "category": "chosen_category",
    "confidence": confidence_score,
    "explanation": "brief explanation"
}"""

    success, result = test_json_response(
        prompt, "You are a medical text classification expert."
    )
    if success:
        print("✅ Test 1 passed - Valid JSON response received")
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"❌ Test 1 failed - Error: {result}")

    # Test 2: Structured data extraction
    print("\nTest 2: Structured Data Extraction")
    schema = {
        "patient_demographics": {"age": "number", "gender": "string"},
        "vital_signs": {"temperature": "number", "heart_rate": "number"},
    }

    prompt = f"""Extract information from the following medical text according to this schema:
{json.dumps(schema, indent=2)}

Text: 45-year-old male presents with fever. Vitals: T 38.2, HR 110.

Provide the extracted data in JSON format matching the schema exactly.
If a field is not found in the text, use null as the value."""

    success, result = test_json_response(
        prompt, "You are a medical information extraction expert."
    )
    if success:
        print("✅ Test 2 passed - Valid JSON response received")
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"❌ Test 2 failed - Error: {result}")

    # Test 3: Medical entity extraction
    print("\nTest 3: Medical Entity Extraction")
    prompt = """Extract all medical entities from the following text:
    
Text: Patient with history of hypertension and type 2 diabetes. Recent CBC shows elevated WBC. Taking lisinopril and metformin.

Categorize them into:
- Conditions
- Medications
- Procedures
- Lab tests
- Vital signs

Return the results in JSON format."""

    success, result = test_json_response(
        prompt, "You are a medical NLP expert."
    )
    if success:
        print("✅ Test 3 passed - Valid JSON response received")
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"❌ Test 3 failed - Error: {result}")

    # Summary
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
