# Assignment 8: Murder Mystery Agents

Build detective agent workflows that solve two murder mysteries using the OpenAI Agents SDK and **prompt chaining**. Each agent handles one stage of the investigation and passes structured results to the next.

## Mysteries

- **Part 1: Murder at the Mountain Cabin** : A narrative mystery with LLM-powered suspects. Your agents search locations, interrogate characters, and deduce the killer across three stages.
- **Part 2: Death at St. Mercy Hospital** : A structured logic puzzle. Your agents gather evidence and use deduction to identify the killer, weapon, and time of death.

## Setup

### API Key

You need an OpenRouter API key (or OpenAI key) for this assignment. Get the shared key from the class forum, or create a free account at [openrouter.ai](https://openrouter.ai).

Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
```

**WARNING**: Do not commit your `.env` file. GitHub will invalidate it for all students.

## What You'll Do

We're building a detective! Unfortunately, it doesn't quite know how to do its job yet...

Open `assignment.ipynb` and write the `instructions` prompt for **5 pre-defined agents** across 4 TODOs. Each Agent is already wired up with its name, tools, and output type. Your job is to write the prompt that tells it what to do.

**Part 1 - Cabin Mystery (3 agents):**

| TODO | Agent | The agent needs to know... |
| ---- | ----- | ------------------------- |
| 1 | Crime Scene Investigator | Its role, all six location IDs, and how to summarize findings |
| 2 | Lead Interrogator | The evidence found so far, who/how to interview, and when/how to follow up |
| 3 | Lead Detective | How to weigh physical evidence against alibis and identify contradictions |

**Part 2 - Hospital Mystery (2 agents):**

| TODO | Agent | The agent needs to know... |
| ---- | ----- | ------------------------- |
| 4a | Evidence Collector | Which tools to use, suspects and their IDs, and what to gather |
| 4b | Forensic Analyst | Which evidence can't be faked, how to catch liars, and how to determine timing |

The notebook also includes an **interactive detective** you can use to manually search locations and interrogate suspects. This helpful for understanding the mystery and iterating on your agent instructions when the investigation inevitably goes off the rails.

## Output Files

Tests check these artifacts (generated when you run the notebook):

- `output/part1_results.json` : Part 1 accusation (killer, weapon, motive, evidence, transcript)
- `output/part2_results.json` : Part 2 solution (killer, weapon, time of death, reasoning, transcript)

Tests verify your agent identified the correct answers and actually used their tools.
