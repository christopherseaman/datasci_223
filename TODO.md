# TODO: Lecture 1 Improvements

## Overall Structure Changes
- Reorganize content to include 3 distinct hands-on demos at roughly 1/3, 2/3, and end points:
  1. **Demo 1 (1/3)**: Command line basics and navigation
  2. **Demo 2 (2/3)**: Git and GitHub setup with GitHub Codespaces
  3. **Demo 3 (End)**: Python basics and running a simple script

- Tighten up content to focus on review material
- Remove redundant information and consolidate sections
- Add clear section headers and improve flow between topics

## Specific Content Additions

### 1. Command Line Review Section
- Fill in the empty "Command Line Basics" section with:
  - Terminal access on different platforms (Mac, Windows with WSL, Linux)
  - Basic navigation commands (`pwd`, `ls`, `cd`)
  - File operations (`mkdir`, `touch`, `cp`, `mv`, `rm`)
  - Viewing file contents (`cat`, `head`, `tail`)
  - Text manipulation with `grep` and `nano` (mention of regex, but not deep explanation. link to resource?)
  - Chaining with pipes (`|`) and redirection (`>`)
  - Special directories (`~`, `.`, `..`)
  - Include common flags/options for each command
  - Demo: Add examples for each command with expected output (incl. flags)

### 2. GitHub Codespaces Emphasis
- Expand the GitHub Codespaces section:
  - Detailed instructions on setting up and accessing GitHub Codespaces
  - Screenshots of the interface and key features
  - Highlight extra hours free for students (with GitHub Education)
  - I actually write my lectures on my ipad using Codespaces and VS Code tunnels (privately hosted Codespace, sorta)
  - Explain benefits: pre-configured environment, no local setup needed, consistent experience
  - Add instructions for persisting work between sessions: codespaces don't last forever, but will stick around for weeks. Make a branch and commit/push often to save WIP
  - Include how to install extensions in Codespaces

### 3. GitHub Classroom Introduction
- Add new section on GitHub Classroom:
  - Explain how assignments will be distributed through GitHub Classroom
  - Instructions for accepting assignments
  - Overview of how work will be graded (automated tests, manual review)
  - Submission process and deadlines
  - How to check feedback and grades
  - Benefits of using GitHub Classroom for version control and collaboration

### 4. WSL Over Native Windows
- 
- Enhance the WSL section:
  - Clear step-by-step installation instructions for WSL
  - Explain why WSL is preferred over native Windows (unix consistency)
  - How to access Windows files from WSL (`/mnt/c/...`)
  - How to access WSL files from Windows (`\\wsl$\...`)
  - Terminal options for WSL (Windows Terminal, VS Code integrated terminal)
  - Common troubleshooting tips

### 5. Python Environment Setup
- Revise the Python installation section:
  - Focus on consistent setup across platforms
  - For Windows, emphasize Python installation through WSL
  - Add more details on virtual environments and their importance
  - Include information on `.gitignore` for Python projects

## Content to Remove or Condense
- Reduce the extensive Markdown explanation (keep it more concise)
- Condense the "It came from the Internet" section or move to supplementary material
- Move the data cleaning plan reference to the next week as noted in the FIXME comment
- Remove or condense the "Additional options" section to focus on the recommended path

## Demo Outlines

### Demo 1: Command Line Basics
- Show terminal access on different platforms
- Demonstrate basic navigation and file operations
- Create a directory structure for a simple project
- Show how to view and manipulate file contents
- Interactive component: Have students follow along with basic commands

### Demo 2: Git and GitHub with Codespaces
- Demonstrate GitHub account setup and GitHub Education benefits
- Show how to create a repository and clone it
- Introduce GitHub Codespaces and how to launch it
- Make changes, commit, and push from Codespaces
- Demonstrate a simple branch and merge workflow

### Demo 3: Python Basics
- Show how to run Python in different environments (local, Codespaces, Jupyter)
- Demonstrate basic Python syntax and data types
- Create and run a simple Python script
- Show how to use virtual environments
- Connect to the assignment for the week

## Assignment Updates
- Update the assignment section to use GitHub Classroom
- Create an autograded Python assignment as noted in the FIXME comment
- Add specific learning objectives for the assignment
- Provide clearer instructions for the README.md creation. Should be auto-gradable on GH Classroom, so something like:
  - Write a python script that hashes the first argument you give it using XXX
  - Run the script with your email address as the argument and redirect (add to command line section) the output of the script to 'hash.email'. Auto-grader with check against email listing 
  - README.md with little about what they are hoping to get out of the course and any topics they'd like to see included (help choose future lecture topics)
  - README.md with a musical recommendation and a link to something about it (could be Youtube/Spotify, Wikipedia, band website, anything) (see basic markdown formatting)