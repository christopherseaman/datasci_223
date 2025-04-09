Edit @/lectures/02/lecture_02.md  to better match the following instructions, including but not limited to:

- Recommending topics to add, expand, or remove given the audience, topic, and time constraints
- Adding visual cues and visual learning content or suggestions (e.g., add an image link for a screenshot that doesn't exist)
- Talking points in HTML comments. These should include the content to be shared (not directions for what the instructor should talk about) so that they are valuable to students reviewing the material on their own
- Balance of content: each topic should include conceptual, reference, and example material
- Demos match to lecture content: the demos should demonstrate the content clearly and not require or reference additional material outside the lecture @/lectures/02/demo/demo1-print-debugging.md @/lectures/02/demo/demo2-pdb-debugging.md @/lectures/02/demo/demo3-vscode-debugging.md @/lectures/02/demo/demo4-bigdata.md 
- @/lectures/02/assignment/README.md @/lectures/02/assignment/patient_data_cleaner.py @/lectures/02/assignment/med_dosage_calculator.py  match to lecture content: the assignment should be doable using the tools provided in lecture; the assignment should be designed for autograding via github action unit tests

----

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
- **Visual Learning:** Use diagrams, analogies, screenshots, and concrete examples/outputs
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
p