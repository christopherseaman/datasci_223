# Assignment 8 Hints

## The Iteration Loop

The core skill in this assignment is **prompt engineering through iteration**. You don't need to know the mystery's answer — you need to write instructions that make the agent investigate thoroughly enough to find it. The loop is:

1. Write instructions → run the agent → check the output
2. If the output is wrong or incomplete, look at the intermediate results (the checkpoints help)
3. Adjust your instructions → re-run → check again

Each stage's structured output (`evidence`, `findings`, `hospital_evidence`) is inspectable — print it to see exactly what the next agent will receive.

## TODO 1: Stage 1 — Crime Scene Investigation

The agent definition, tools, and Runner call are already provided. You just write `instructions`. Your instructions should:

- Give the agent a role ("You are a crime scene investigator...")
- **List all six location IDs explicitly**: `living_room`, `kitchen`, `mudroom`, `upstairs_hallway`, `back_porch`, `marcus_bedroom`
- Tell it to summarize what it finds

**How to check:** You should see 6 `display_search` cards (one per location) and `evidence.clues` should have 10+ items. If fewer, your instructions didn't list all locations.

## TODO 2: Stage 2 — Suspect Interrogation

The `evidence_brief` is already embedded in the f-string — you write the rest of the instructions around it. Your instructions should:

- Give the agent a role
- Name all four suspects (`diana`, `larry`, `tom`, `sofia`)
- Tell it to ask about alibis
- Tell it to **follow up** when answers seem evasive or contradict the physical evidence

**How to check:** You should see 4+ chat bubble pairs (initial questions + follow-ups). `findings.contradictions` should NOT be empty — if it is, your instructions aren't directing the agent to press suspects. Try telling it specifically: "Follow up if their answer contradicts physical evidence."

## TODO 3: Stage 3 — Final Deduction

This agent has no tools — it only sees the `case_file` passed in the prompt. Your instructions should tell it **how to reason**:

- Physical evidence is more reliable than what suspects say
- Compare each suspect's alibi against the physical evidence — whose story doesn't add up?
- Look for the murder weapon in the crime scene evidence
- Look for motive in the victim's documents

**How to check:** If the accusation is wrong, `print(case_file)` to see what the deduction agent received. If the contradictions are weak or missing, go back and improve TODO 2's instructions. The deduction agent can only work with what it's given.

## TODO 4: Hospital Mystery (Two Stages)

Same chaining pattern as Part 1 — split into evidence gathering and deduction.

**Stage 1 (Evidence Collector):** Your instructions should tell the agent to use every available tool and get statements from ALL four suspects (`dr_blake`, `nurse_chen`, `dr_santos`, `orderly_james`).

**Stage 2 (Forensic Analyst):** Your instructions should include deduction principles:
- Keycard logs are hardware records — they **cannot be faked**
- Compare each suspect's **statement** against their **keycard activity** — the liar is the killer
- Match the **weapon** to the **cause of death**
- For **time of death**: the victim was still alive at the time of her last activity (email). The murder happened **after** that. Look for a **witness observation** that places someone near the victim's room at a specific time — that's when it happened

## Debugging: If the Agent Gets It Wrong

**Part 1:**
- Print `evidence.clues` — are there clues from all 6 locations?
- Print `findings.contradictions` — did the interrogator find any? If empty, your Stage 2 instructions need to be more specific about what to ask
- Print `case_file` — does it contain enough information for a human to solve it?
- Use the interactive detective to manually question suspects and understand the mystery

**Part 2:**
- Print `hospital_evidence.witness_observations` — does it include a specific time and location?
- Print `hospital_evidence.keycard_logs` — are all entries preserved with times?
- Print `evidence_dossier` — can YOU solve it from reading it? If not, the agent can't either
- The most common error is wrong time of death — that comes from witness observations, not the victim's last email

## Common Issues

- **Module not found**: Run the `%pip install` cell at the top of the notebook
- **API key not found**: Make sure `.env` has `OPENROUTER_API_KEY=...` (no quotes around the value)
- **Agent runs out of turns**: Increase `max_turns` or make your instructions more focused
- **Wrong answer**: Improve your instructions — check intermediate outputs to see where the chain broke down
- **NameError for `evidence`/`findings`/`accusation`/`solution`**: Make sure you uncommented and ran each TODO cell in order
