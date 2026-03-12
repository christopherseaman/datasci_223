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

# Demo 1: Building Clinical Agents

Build agents that use tools, follow instructions, and specialize in different clinical roles.

## Setup

```python
%pip install -q openai openai-agents python-dotenv
```

```python
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

# Load API key from .env file (OPENROUTER_API_KEY or OPENAI_API_KEY)
load_dotenv()

# OpenAIChatCompletionsModel wraps any OpenAI-compatible API endpoint.
# OpenRouter proxies dozens of model providers through one API —
# we point the SDK's async client at it and pick a model by its OpenRouter ID.
if os.environ.get("OPENROUTER_API_KEY"):
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    MODEL = OpenAIChatCompletionsModel(model="openai/gpt-4o-mini", openai_client=client)
    MODEL_NAME = "openai/gpt-4o-mini"
    # Tracing requires OpenAI API — disable for OpenRouter
    set_tracing_disabled(True)
elif os.environ.get("OPENAI_API_KEY"):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)
    MODEL_NAME = "gpt-4o-mini"
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

# temperature=0 for more deterministic output across runs
SETTINGS = ModelSettings(temperature=0, max_tokens=1024)

print(f"Using model: {MODEL_NAME}")
```

## Section 1: Your First Agent

An agent needs **instructions** (what it does) and a **runner** (the loop that calls the model, executes tools, and feeds results back). Instructions are the system prompt that shapes behavior — what the agent focuses on, how it responds, what constraints it follows.

```python
# Agent = instructions + model. Runner = the agent loop that orchestrates everything.
agent = Agent(
    name="Health Assistant",
    model=MODEL,
    model_settings=SETTINGS,
    instructions="You are a concise clinical assistant. Answer health questions clearly and briefly.",
)

# Runner.run() sends the prompt through the agent loop:
# 1. Model receives instructions (system prompt) + user message
# 2. Model generates a response (or calls tools — none here)
# 3. Runner returns the result
result = await Runner.run(agent, "What are the warning signs of a heart attack?")
print(result.final_output)
```

That's the simplest possible agent — an LLM with a defined role, run through the SDK's agent loop. Not much different from a raw API call yet. The `Agent` abstraction lets us layer on tools and structure.

## Section 2: Tool Use

An agent becomes useful when it can *do* things, not just *say* things. `@function_tool` turns a Python function into a tool the agent can call. The agent decides *when* to use each tool based on the conversation — that's the "act" step in the agent loop.

This is critical for clinical calculations: **math happens in Python, not in the LLM**. The model extracts values from the prompt, the tools compute, and the model reports results. This avoids the arithmetic errors LLMs are prone to.

```python
# @function_tool exposes a Python function to the agent.
# The SDK reads the function's type hints and docstring to generate a JSON schema
# that tells the model what the tool does and what arguments it expects.

@function_tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate BMI from weight in kg and height in meters."""
    bmi = weight_kg / (height_m ** 2)
    category = (
        "underweight" if bmi < 18.5
        else "normal weight" if bmi < 25
        else "overweight" if bmi < 30
        else "obese"
    )
    return f"BMI: {bmi:.1f} ({category})"


@function_tool
def calculate_creatinine_clearance(
    creatinine: float, age: int, weight_kg: float, is_female: bool
) -> str:
    """Estimate creatinine clearance using the Cockcroft-Gault equation."""
    crcl = ((140 - age) * weight_kg) / (72 * creatinine)
    if is_female:
        crcl *= 0.85
    return f"CrCl: {crcl:.0f} mL/min"
```

```python
# tools= gives the agent access to our functions. The model sees their schemas
# and decides which to call (and with what arguments) based on the user's message.
clinical_calculator = Agent(
    name="Clinical Calculator",
    model=MODEL,
    model_settings=SETTINGS,
    instructions=(
        "You are a clinical assistant with access to medical calculators. "
        "Use the available tools to compute values — never do arithmetic yourself. "
        "Report results clearly with clinical context."
    ),
    tools=[calculate_bmi, calculate_creatinine_clearance],
)

# The agent loop now has real work to do:
# 1. Model reads the prompt, decides it needs both tools
# 2. Model generates tool calls with extracted arguments (82kg, 1.68m, etc.)
# 3. Runner executes the Python functions, feeds results back to the model
# 4. Model composes a final response incorporating both tool outputs
result = await Runner.run(
    clinical_calculator,
    "Patient is 82 kg, 1.68 m tall. Creatinine 1.4 mg/dL, age 68, male. "
    "Calculate BMI and creatinine clearance.",
)
print(result.final_output)
```

The agent chose which tools to call, extracted the right arguments from the prompt, and composed the results into a clinical summary. The tool definitions (type hints + docstrings) are all the model needs to know what's available and how to call it.

## Section 3: Specialized Agents

Different agents can have different **instructions** that shape *what* they focus on and *how* they reason. Same patient data, different clinical lens — the instructions act as each agent's specialization.

```python
# Three agents, same model, different instructions.
# Each one reads the same patient data but focuses on different aspects.

diagnostician = Agent(
    name="Diagnostician",
    model=MODEL,
    model_settings=SETTINGS,
    instructions=(
        "You are a diagnostician. Given patient information, identify the most likely "
        "diagnosis, generate a differential, and recommend workup. Think systematically "
        "through the clinical presentation. Be concise."
    ),
)

pharmacist = Agent(
    name="Clinical Pharmacist",
    model=MODEL,
    model_settings=SETTINGS,
    instructions=(
        "You are a clinical pharmacist. Given a patient presentation, recommend "
        "medications with specific dosing, flag contraindication concerns given the "
        "patient's comorbidities and current medications, and outline what to monitor. "
        "Be concise and practical."
    ),
)

summarizer = Agent(
    name="Chart Summarizer",
    model=MODEL,
    model_settings=SETTINGS,
    instructions=(
        "You summarize clinical encounters into a chart note. Produce a one-liner summary, "
        "list active problems, and outline the plan. Be very concise — this goes in the chart."
    ),
)

print("Defined 3 specialized agents:")
for a in [diagnostician, pharmacist, summarizer]:
    print(f"  - {a.name}")
```

Same patient, three different specialists. Each agent focuses on what its instructions tell it to — the diagnostician thinks about differential diagnoses, the pharmacist thinks about drug interactions, and the summarizer distills everything down.

```python
# One complex patient presentation with multiple comorbidities.
# Each agent will extract different signals from the same data.
patient = (
    "72-year-old male presenting with increasing shortness of breath over 3 days. "
    "History of COPD, type 2 diabetes on metformin 1000mg BID, hypertension on "
    "lisinopril 20mg daily. Vitals: BP 158/92, HR 96, SpO2 89% on room air. "
    "Chest X-ray shows bilateral infiltrates. WBC 14.2, BNP 890."
)

dx_result = await Runner.run(diagnostician, "Assess this patient:\n" + patient)
print("DIAGNOSTICIAN:")
print(dx_result.final_output)
```

```python
# Same patient data, but the pharmacist's instructions direct its attention
# to medications, contraindications, and monitoring — not diagnosis.
rx_result = await Runner.run(pharmacist, "Assess this patient:\n" + patient)
print("PHARMACIST:")
print(rx_result.final_output)
```

```python
summary_result = await Runner.run(summarizer, "Summarize this encounter:\n" + patient)
print("CHART SUMMARY:")
print(summary_result.final_output)
```

Three agents, same patient data, three different perspectives. The diagnostician ignores medication dosing; the pharmacist ignores differential diagnosis; the summarizer distills everything into a chart note. All of this is driven by instructions alone — no code changes between them, just different system prompts.
