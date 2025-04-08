## Meta-Instruction

- **Audience Assessment:** Create content for health data science masters students who are beginners in programming (Python, git, shell commands)
- **Continuous Evaluation:** Before generating each section, ensure content matches student level, balances concept / reference / example, and includes speaking notes
- **Time Structure:** Design for 90-minute lectures, with maximum 60 slides if using Marp. Otherwise, long-form Markdown.
- **Demo Integration:** Include 3 hands-on demo breaks (at ⅓, ⅔, and end points)

## Core Principles

### 1. Format & Structure

- **Markdown Format:** Create slides in marp markdown format (separated by `---`) or long-form Markdown, as requested.
- **Speaking Notes:** Include detailed instructor notes in HTML comments with specific explanations
- **Progressive Learning:** Structure content to build knowledge incrementally
- **Visual Organization:** Use consistent heading levels, bullet points, and white space

### 2. Content Balance

- **Conceptual Foundations:** Explain how things work in accessible terms
- **Reference Material:** Include function definitions, syntax rules, and common parameters
- **Practical Examples:** Provide executable code samples with health data applications
- **Health Relevance:** Connect Python concepts to health data scenarios whenever possible

### 3. Teaching Approach

- **Beginner-Friendly:** Avoid jargon, explain terms when introduced
- **Visual Learning:** Use diagrams, analogies, and concrete examples
- **Engagement:** Include comprehension checkpoints and practice opportunities
- **Misconceptions:** Address common beginner mistakes in speaking notes

### 4. Tone & Style

- **Professional but Engaging:** Maintain educational focus while being approachable
- **Strategic Humor:** Include occasional nerdy puns (xkcd-style) and cheesy pop culture references (80s/90s movies)
- **Visual Cues:** Use emoji and formatting to highlight key points and create visual interest
- **Clear Annotations:** Comment key lines within code examples

### 5. Demo Break Structure

- **Hands-On Learning:** Design 3 practical demo sessions (10-15 minutes each)
- **Progressive Difficulty:** Start simple, build complexity across demos
- **Clear Instructions:** Provide step-by-step guidance with expected outcomes
- **Success Validation:** Include ways to confirm students completed tasks correctly

----

create long-form markdown lecture notes (projected/shared instead of slides) based on @/lectures/02/outline.md and look up details as needed based on content locations in @/refs/debugging/inventory.md . Target is 90 minute lecture (including demos) with 45-60 min spent on debugging Let's make the lecture notes in parts, in the lectures/02/parts folder have 
1. A .md for the Jupyter part
2. Folder for debugging with a .md for each major topic plus .py scripts and .md's for demos/assignment
3. Folder for working with data bigger than memory with a .md for the notes and .py scripts and .md's for demos/assignment