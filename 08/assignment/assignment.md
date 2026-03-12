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

# Assignment 8: Murder Mystery Agents

Build detective agent workflows that solve two murder mysteries using the OpenAI Agents SDK. You'll use **prompt chaining** — the workflow pattern from the lecture — where each agent handles one stage of the investigation and passes structured results to the next.

**What you'll do:** Each Agent is already defined with its name, model, tools, and output type — you write the `instructions` prompt that tells it what role to play, what to investigate, and how to reason about the evidence. That's it — **5 prompts across 4 TODOs**. Everything else is provided.

## Setup

```python
%pip install -q -r requirements.txt
```

```python
%reset -f

import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from IPython.display import display, HTML
from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

load_dotenv()
os.makedirs("output", exist_ok=True)

# Same setup pattern as the demos: explicit model object + settings
set_tracing_disabled(True)

if os.environ.get("OPENROUTER_API_KEY"):
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    AGENTS_MODEL = OpenAIChatCompletionsModel(model="openai/gpt-4o-mini", openai_client=client)
elif os.environ.get("OPENAI_API_KEY"):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    AGENTS_MODEL = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client)
else:
    raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

SETTINGS = ModelSettings(temperature=0, max_tokens=4096)

print("Setup complete")
```

## Display Helpers

These functions render investigation output as styled HTML — evidence cards, chat bubbles, detective notes, and verdict panels. You don't need to modify these; they're called automatically by the tools and checkpoint cells.

```python
from helpers import (
    case_briefing, display_stage, display_search, display_interrogation,
    display_notes, display_witness, display_verdict,
)
```

## Transcript Logger

Tracks what the detective agents do — which tools they call and what they learn. The transcript is saved with results for grading.

```python
transcript = []

def log_action(action_type: str, detail: str):
    """Append an entry to the investigation transcript."""
    transcript.append({"action": action_type, "detail": detail})
```

---

# Part 1: Murder at the Mountain Cabin

A group of old college friends reunited at a remote mountain cabin. By Saturday morning, the host — Marcus Reed — was found dead. The cabin is snowed in. The killer is among the guests.

You'll build a **three-stage investigation workflow** using prompt chaining:

1. **Crime Scene Investigation** — An agent searches all locations and summarizes the evidence
2. **Suspect Interrogation** — An agent uses the evidence to conduct targeted interviews
3. **Final Deduction** — An agent analyzes everything and makes the accusation

Each stage produces structured output (`output_type`) that feeds into the next — the same chaining pattern from Demo 3.

```python
with open("mystery_data.json") as f:
    cabin_data = json.load(f)

case_briefing(
    title=cabin_data["title"],
    setting=cabin_data["setting"],
    victim_name=cabin_data["victim"]["name"],
    cause=cabin_data["victim"]["cause_of_death"],
    suspects=[c["name"] + " — " + c["role"] for c in cabin_data["characters"].values()],
    locations=[loc["name"] for loc in cabin_data["locations"].values()],
)
```

### Part 1 Tools

These tools let agents interact with the mystery world. `search_location` returns clues from a location. `interrogate` creates a temporary sub-agent for each suspect — the suspect has its own personality and secrets embedded in a system prompt, so it responds in character.

```python
@function_tool
def search_location(location_id: str) -> str:
    """Search a location in the cabin for clues. Valid locations: living_room, kitchen, mudroom, upstairs_hallway, back_porch, marcus_bedroom"""
    location = cabin_data["locations"].get(location_id)
    if not location:
        return f"Unknown location '{location_id}'. Valid: {', '.join(cabin_data['locations'].keys())}"
    log_action("search", f"Searched {location['name']}")
    display_search(location["name"], location["clues"])
    result = f"**{location['name']}**: {location['description']}\n\nClues found:\n"
    for clue in location["clues"]:
        result += f"- {clue}\n"
    return result


@function_tool
async def interrogate(character_name: str, question: str) -> str:
    """Interrogate a suspect by name. Ask them a specific question. Valid names: diana, larry, tom, sofia"""
    key = character_name.lower().strip()
    character = cabin_data["characters"].get(key)
    if not character:
        return f"Unknown character '{character_name}'. Valid: {', '.join(cabin_data['characters'].keys())}"

    log_action("interrogate", f"Asked {character['name']}: {question[:80]}")

    system_prompt = f"""You are {character['name']}, {character['role']} in a murder investigation.

Personality: {character['personality']}
Background: {character['background']}
Your secret: {character['secret']}
Your alibi: {character['alibi']}

Things you know or observed:
{chr(10).join('- ' + k for k in character['knows'])}

Behavioral guidelines: {character['guardrails']}

Stay in character. Answer the detective's question based on your knowledge, personality, and secrets. Keep responses to 2-4 sentences. You may lie or deflect according to your behavioral guidelines."""

    suspect_agent = Agent(
        name=character["name"],
        model=AGENTS_MODEL,
        model_settings=SETTINGS,
        instructions=system_prompt,
    )
    result = await Runner.run(suspect_agent, question)
    response = result.final_output
    log_action("response", f"{character['name']}: {response[:100]}")
    display_interrogation(character["name"], question, response)
    return response

cabin_tools = [search_location, interrogate]
print(f"Tools ready: {[t.name for t in cabin_tools]}")
```

### Interactive Detective

Use this to manually explore the mystery — search locations and interrogate suspects yourself. This helps you understand what evidence exists before writing agent instructions, and lets you verify your agents found the right things.

```python
interactive_detective = Agent(
    name="Interactive Detective",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="You are assisting a human detective investigating the murder of Marcus Reed at a mountain cabin. The human will tell you who to interrogate or where to search. Use your tools to carry out their requests and report back what you find.",
    tools=cabin_tools,
)

async def investigate(prompt):
    """Quick helper: ask the detective to do something and see the result."""
    result = await Runner.run(interactive_detective, prompt)
    print(f"\n{result.final_output}\n")
```

Try it out — run these to explore the mystery before writing your agent instructions:

```python
# await investigate("Search the living room")
```

```python
# await investigate("Ask Larry about his alibi")
```

### Part 1 Structured Outputs

Each stage of the investigation produces structured data that feeds into the next. The `output_type` parameter forces each agent to return a Pydantic model instead of free text.

```python
class EvidenceSummary(BaseModel):
    """Stage 1 output: what the crime scene investigation found."""
    clues: list[str]
    suspicious_items: list[str]
    key_questions: list[str]

class InterviewFindings(BaseModel):
    """Stage 2 output: what the interrogations revealed."""
    alibis: list[str]
    contradictions: list[str]
    key_observations: list[str]

class Accusation(BaseModel):
    """Stage 3 output: the final accusation."""
    killer: str
    weapon: str
    motive: str
    evidence: list[str]
```

### TODO 1: Stage 1 — Crime Scene Investigation

Write `instructions` for an agent that searches all six locations and summarizes the evidence. Your instructions should tell the agent its role, list all six location IDs to search, and ask it to summarize what it finds.

**Location IDs:** `living_room`, `kitchen`, `mudroom`, `upstairs_hallway`, `back_porch`, `marcus_bedroom`

```python
display_stage(1, "Crime Scene Investigation", "Searching all locations for physical evidence and clues")
transcript = []  # Reset transcript for Part 1

search_agent = Agent(
    name="Crime Scene Investigator",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="",  # TODO: Write your instructions here
    tools=[search_location],
    output_type=EvidenceSummary,
)

# Uncomment to run your agent:
# search_result = await Runner.run(search_agent, "Search all six locations in the cabin for evidence.", max_turns=15)
# evidence = search_result.final_output
```

```python
# Checkpoint: see what your agent found (uncomment after running TODO 1)
# display_notes("Crime Scene Summary", {
#     f"Clues found ({len(evidence.clues)})": evidence.clues,
#     "Suspicious items": evidence.suspicious_items,
#     "Questions to investigate": evidence.key_questions,
# })
```

**Checkpoint:** You should see 6 search cards (one per location) and 10+ clues. If the agent missed locations, make your instructions more explicit. Use `await investigate("Search the kitchen")` above to manually verify what's in each location.

### TODO 2: Stage 2 — Suspect Interrogation

Write `instructions` for an agent that uses the evidence from Stage 1 to interrogate suspects. The key pattern: **embed Stage 1's output in Stage 2's instructions** — this is how results flow between agents in a chain.

The `evidence_brief` below formats Stage 1's structured output as text. Your instructions should include it (via f-string) so the interrogator knows what to ask about.

```python
display_stage(2, "Suspect Interrogation", "Questioning suspects based on crime scene evidence")

# This formats Stage 1 output as text for the interrogator's instructions
evidence_brief = f"""Evidence found so far:
Clues: {chr(10).join('- ' + c for c in evidence.clues)}
Suspicious items: {', '.join(evidence.suspicious_items)}
Key questions: {chr(10).join('- ' + q for q in evidence.key_questions)}"""

interview_agent = Agent(
    name="Lead Interrogator",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions=f"""

{evidence_brief}

""",  # TODO: Add your interrogation strategy around the evidence_brief above
    tools=[interrogate],
    output_type=InterviewFindings,
)

# Uncomment to run your agent:
# interview_result = await Runner.run(interview_agent, "Interrogate all four suspects.", max_turns=25)
# findings = interview_result.final_output
```

```python
# Checkpoint: see what your interrogator found (uncomment after running TODO 2)
# display_notes("Interrogation Summary", {
#     "Alibis": findings.alibis,
#     "Contradictions": findings.contradictions,
#     "Key observations": findings.key_observations,
# })
```

**Checkpoint:** You should see chat bubbles for each suspect and at least one contradiction in the findings. If `contradictions` is empty, your instructions need to tell the agent to follow up when answers seem evasive or contradict physical evidence. Use `await investigate("Ask Larry about his muddy boots")` to manually explore what suspects say.

### TODO 3: Stage 3 — Final Deduction

Write `instructions` for an agent that analyzes ALL evidence and makes the final accusation. This agent has **no tools** — it just reasons over the `case_file` built from Stages 1 and 2.

```python
display_stage(3, "Final Deduction", "Analyzing all evidence to identify the killer")

# This combines all evidence for the deduction agent
case_file = f"""{evidence_brief}

Interview findings:
Alibis: {chr(10).join('- ' + a for a in findings.alibis)}
Contradictions: {chr(10).join('- ' + c for c in findings.contradictions)}
Key observations: {chr(10).join('- ' + o for o in findings.key_observations)}"""

deduction_agent = Agent(
    name="Lead Detective",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="",  # TODO: Write your instructions here
    output_type=Accusation,
)

# Uncomment to run your agent:
# deduction_result = await Runner.run(deduction_agent, f"Based on all evidence, make your accusation:\n\n{case_file}", max_turns=3)
# accusation = deduction_result.final_output
```

```python
# Uncomment to see the verdict:
# display_verdict("ACCUSATION", {
#     "Killer": accusation.killer,
#     "Weapon": accusation.weapon,
#     "Motive": accusation.motive,
#     "Evidence": ", ".join(accusation.evidence),
# })
```

**Checkpoint:** If the accusation is wrong, the deduction agent can only reason about what's in `case_file`. Print it (`print(case_file)`) to see if the evidence and contradictions are specific enough. If not, go back and improve your Stage 2 instructions. This iterative refinement is how real agent development works.

**Hint:** Your instructions should tell the agent to compare each suspect's alibi against the physical evidence — physical evidence doesn't lie (muddy boots, wiped fingerprints, moved objects). The suspect whose alibi is most clearly contradicted is the killer.

### Save Part 1 Results

```python
part1_results = {
    "killer": accusation.killer,
    "weapon": accusation.weapon,
    "motive": accusation.motive,
    "evidence": accusation.evidence,
    "transcript": transcript,
}

with open("output/part1_results.json", "w") as f:
    json.dump(part1_results, f, indent=2)

print(f"Saved to output/part1_results.json ({len(transcript)} transcript entries)")
```

---

# Part 2: Death at St. Mercy Hospital

A structured logic puzzle. Dr. Eleanor Voss was found dead in her office. Security footage shows no one entered or left overnight — the killer was already inside. Use deduction tools to analyze evidence and identify the killer, weapon, and time of death.

```python
with open("mystery_o_matic.json") as f:
    hospital_data = json.load(f)

case_briefing(
    title=hospital_data["title"],
    setting=hospital_data["setting"],
    victim_name=hospital_data["victim"]["name"],
    cause=hospital_data["victim"]["cause_of_death"],
    suspects=[s["name"] + " — " + s["role"] for s in hospital_data["suspects"].values()],
)
```

### Part 2 Tools

Deduction tools that give the agent access to different categories of evidence. Unlike Part 1, there are no interactive sub-agents — just data retrieval tools.

```python
@function_tool
def get_room_map() -> str:
    """Get the hospital floor map showing all rooms and their locations."""
    log_action("tool", "Retrieved room map")
    result = "Hospital Floor Map:\n"
    for floor, rooms in hospital_data["rooms"].items():
        result += f"\n{floor.replace('_', ' ').title()}: {', '.join(rooms)}"
    return result


@function_tool
def get_witness_statement(suspect: str) -> str:
    """Get a suspect's statement and background. Valid suspects: dr_blake, nurse_chen, dr_santos, orderly_james"""
    key = suspect.lower().strip().replace(" ", "_").replace("dr.", "dr").replace("dr ", "dr_")
    suspect_data = hospital_data["suspects"].get(key)
    if not suspect_data:
        return f"Unknown suspect '{suspect}'. Valid: {', '.join(hospital_data['suspects'].keys())}"
    log_action("interview", f"Got statement from {suspect_data['name']}")
    display_witness(suspect_data['name'], suspect_data['role'], suspect_data['statement'])
    return f"""**{suspect_data['name']}** ({suspect_data['role']})
Motive: {suspect_data['motive']}
Statement: "{suspect_data['statement']}" """


@function_tool
def get_evidence() -> str:
    """Get all physical evidence and keycard logs from the investigation."""
    log_action("tool", "Retrieved evidence and keycard logs")
    result = "=== KEYCARD ACCESS LOGS ===\n"
    for entry in hospital_data["evidence"]["keycard_logs"]:
        result += f"{entry['time']} | {entry['person']} | {entry['location']} ({entry['action']})\n"
    result += "\n=== PHYSICAL EVIDENCE ===\n"
    for item in hospital_data["evidence"]["physical_evidence"]:
        result += f"- {item}\n"
    result += "\n=== WITNESS OBSERVATIONS ===\n"
    for obs in hospital_data["evidence"]["witness_statements"]:
        result += f"- {obs['witness']}: {obs['statement']}\n"
    return result


@function_tool
def get_weapons() -> str:
    """Get information about potential murder weapons found or available in the hospital."""
    log_action("tool", "Retrieved weapons analysis")
    result = "=== WEAPONS ANALYSIS ===\n"
    for weapon_id, weapon in hospital_data["weapons"].items():
        result += f"\n**{weapon_id.replace('_', ' ').title()}**: {weapon['description']}\n"
        result += f"  Availability: {weapon['availability']}\n"
        result += f"  Notes: {weapon['notes']}\n"
    return result


@function_tool
def get_rules() -> str:
    """Get the logical rules and constraints for solving this mystery."""
    log_action("tool", "Retrieved investigation rules")
    result = "=== INVESTIGATION RULES ===\n"
    for i, rule in enumerate(hospital_data["rules"], 1):
        result += f"{i}. {rule}\n"
    return result

hospital_tools = [get_room_map, get_witness_statement, get_evidence, get_weapons, get_rules]
print(f"Tools ready: {[t.name for t in hospital_tools]}")
```

### Part 2 Structured Outputs

```python
class HospitalEvidence(BaseModel):
    """Stage 1 output: all evidence gathered from the hospital."""
    rules: list[str]
    keycard_logs: list[str]
    witness_observations: list[str]
    suspect_statements: list[str]
    physical_evidence: list[str]
    weapons_info: str

class PuzzleSolution(BaseModel):
    """Stage 2 output: the final solution."""
    killer: str
    weapon: str
    time_of_death: str
    reasoning: str
```

### TODO 4: Solve the Hospital Mystery

Same two-stage pattern as Part 1: first gather all evidence, then deduce the answer.

**Stage 1:** Write `instructions` for an evidence collector that uses ALL five tools to gather everything — room map, rules, keycard logs, physical evidence, weapons, and all four suspect statements.

**Stage 2:** Write `instructions` for a solver that receives the evidence dossier and deduces the answer. No tools — just reasoning.

```python
display_stage(1, "Evidence Collection", "Gathering all physical evidence, logs, and witness statements")
transcript = []  # Reset transcript for Part 2

evidence_collector = Agent(
    name="Evidence Collector",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="",  # TODO: Write your instructions here
    tools=hospital_tools,
    output_type=HospitalEvidence,
)

# Uncomment to run your agent:
# collect_result = await Runner.run(evidence_collector, "Collect all evidence for the Dr. Voss murder case.", max_turns=15)
# hospital_evidence = collect_result.final_output
```

```python
# Checkpoint: see what evidence was collected (uncomment after running Stage 1)
# display_notes("Evidence Collected", {
#     f"Rules ({len(hospital_evidence.rules)})": hospital_evidence.rules,
#     f"Keycard logs ({len(hospital_evidence.keycard_logs)})": hospital_evidence.keycard_logs,
#     f"Witness observations ({len(hospital_evidence.witness_observations)})": hospital_evidence.witness_observations,
#     f"Suspect statements ({len(hospital_evidence.suspect_statements)})": hospital_evidence.suspect_statements,
#     f"Physical evidence ({len(hospital_evidence.physical_evidence)})": hospital_evidence.physical_evidence,
# })
```

```python
display_stage(2, "Forensic Analysis", "Deducing killer, weapon, and time of death from evidence")

# This formats Stage 1 output as text for the solver
evidence_dossier = f"""=== INVESTIGATION RULES ===
{chr(10).join('- ' + r for r in hospital_evidence.rules)}

=== KEYCARD ACCESS LOGS ===
{chr(10).join('- ' + l for l in hospital_evidence.keycard_logs)}

=== WITNESS OBSERVATIONS ===
{chr(10).join('- ' + w for w in hospital_evidence.witness_observations)}

=== SUSPECT STATEMENTS ===
{chr(10).join('- ' + s for s in hospital_evidence.suspect_statements)}

=== PHYSICAL EVIDENCE ===
{chr(10).join('- ' + e for e in hospital_evidence.physical_evidence)}

=== WEAPONS ===
{hospital_evidence.weapons_info}"""

solver = Agent(
    name="Forensic Analyst",
    model=AGENTS_MODEL,
    model_settings=SETTINGS,
    instructions="",  # TODO: Write your instructions here
    output_type=PuzzleSolution,
)

# Uncomment to run your agent:
# solve_result = await Runner.run(solver, f"Analyze this evidence and solve the murder:\n\n{evidence_dossier}", max_turns=3)
# solution = solve_result.final_output
```

```python
# Uncomment to see the verdict:
# display_verdict("SOLUTION", {
#     "Killer": solution.killer,
#     "Weapon": solution.weapon,
#     "Time of Death": solution.time_of_death,
#     "Reasoning": solution.reasoning,
# })
```

**Checkpoint:** The test checks killer, weapon, AND time of death. If any are wrong, print `evidence_dossier` to see what the solver received. Key hints for your solver instructions:
- Keycard logs are hardware records — they CANNOT be faked
- Compare each suspect's statement against the keycard logs — the liar is the killer
- Match the weapon to the cause of death
- Time of death comes from a **witness observation** (what someone saw and when), NOT from the victim's last email

### Save Part 2 Results

```python
part2_results = {
    "killer": solution.killer,
    "weapon": solution.weapon,
    "time_of_death": solution.time_of_death,
    "reasoning": solution.reasoning,
    "transcript": transcript,
}

with open("output/part2_results.json", "w") as f:
    json.dump(part2_results, f, indent=2)

print(f"Saved to output/part2_results.json ({len(transcript)} transcript entries)")
```

---

# Validation

Run the tests to check your results:

```python
!python -m pytest .github/tests/ -v
```

---

# Interactive Interrogation (Optional)

Use the interactive detective to manually explore the cabin mystery. This is helpful for understanding the mystery before writing agent instructions, or for debugging when your agents get the wrong answer.

```python
print("=== Interactive Detective Mode ===")
print("Tell the detective who to interrogate or where to search.")
print("Examples: 'Ask Diana about her alibi', 'Search the kitchen'")
print("Type 'q' to exit.\n")

try:
    while True:
        user_input = input('Detective ("q" to quit)> ').strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Case closed.")
            break
        if not user_input:
            continue
        result = await Runner.run(interactive_detective, user_input)
        print(f"\n{result.final_output}\n")
except (EOFError, KeyboardInterrupt):
    print("\nCase closed.")
except Exception:
    print("(Interactive mode requires a live notebook — skipping in headless execution.)")
```
