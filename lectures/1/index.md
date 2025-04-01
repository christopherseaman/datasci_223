It doesn't matter which tools you use; python and R (and other specialized tools) are quite capable. Since python and R are the most commonly used tools, knowing one or both of them will make it easier to play well with others. Don't try to be an expert in everything! Figure out which you prefer and learn to be "fluent" (able to code a solution from start to finish) in one, then you can get by being "conversational" (able to read and edit others' code) in the other.

Additionally, collaboration usually happens in git and documentation will use markdown. Luckily, those are easy to pick up.

# `git init`

- Tools: `python`, `R`, and `git`
    - Getting set up locally
    - Cloud options (GitHub Codespaces, Colab, Binder, Paperspace)
- Command Line Basics
    - Terminal access on different platforms
    - Basic navigation and file operations
    - Viewing and manipulating file contents
    - Chaining commands with pipes and redirection
- Markdown
    - Syntax summaries
    - GitHub and Notion flavors
    - Readme.md - make one for every repo
- git and GitHub
    - Starting or cloning a repository
    - Git push/pull/sync
    - Branches
    - Conflicts
- Python
    - Syntax basics
    - Running python and jupyter
    - Variables and control flow
    - Common packages
- Runtime Environments
    - Virtual environments
    - Jupyter Notebooks
    - Google Colab (no PHI in this course)
    - GitHub Codespaces (sign up for free student membership)

# Installing tools

For most roles, data science happens in `python` and `R`

## Quickstart

_Note: this is also included in the week's assignment_

These are the standard options that I'll be using to demonstrate going forward. They will also give us a common base to work from, so we can focus on the work rather than tweaking/fixing our development environment.

- Sign up for an account on [GitHub](https://github.com)
  - Apply for [GitHub Education](https://education.github.com/pack) to get extra free hours on Codespaces and other benefits
- Install Python 3 ([instructions](https://docs.python-guide.org))
- Get [VisualStudio Code](https://code.visualstudio.com)
    
    - Most commands are accessed using the "Command Palette"
        - **Shift + Command + P** (Mac)
        - **Ctrl + Shift + P** (Windows/Linux)
        - **F1** (All)
    
    - Extensions
        - Python + Jupyter (use notebooks within VS Code)
        - GitHub Repositories + Remote Repositories (manage git in VS Code instead of the terminal)

**Note:** If you don't want to install software locally, you can use GitHub Codespaces (recommended) or [Google Colab](http://colab.research.google.com) but _never_ use PHI data with public-facing tools.

## GitHub Codespaces

Cloud-based development environment with VS Code in your browser:

- **Benefits**: No setup, consistent environment, works on any device
- **Student perks**: Extra free hours with GitHub Education
- **Getting started**: Repository → Code button → Codespaces tab → Create
- **Persistence**: Codespaces last weeks but not forever; commit/push often
- **Fun fact**: I write these lectures on my iPad using Codespaces and VS Code tunnels

## GitHub Classroom

How we'll manage assignments in this course:

- **Benefits**: Automated distribution, testing, and grading; private repos
- **Process**: Get link → Accept assignment → Clone repo → Make changes → Push to submit
- **Grading**: Automated tests run on submission; feedback via issues/comments

## Command Line Basics

Essential commands for navigating and working with files:

- **Navigation**: `pwd` (where am I?), `ls` (what's here?), `cd` (change directory)
- **Special directories**: `~` (home), `.` (current), `..` (parent)
- **File operations**: `mkdir`, `touch`, `cp`, `mv`, `rm` (careful - no undo!)
- **Viewing content**: `cat`, `head`, `tail`
- **Text tools**: `grep` (search), `nano` (edit)
- **Chaining**: `|` (pipe output), `>` (redirect to file), `>>` (append to file)

Access via Terminal (Mac), WSL (Windows, recommended), or Terminal (Linux)

## Windows Subsystem for Linux (WSL)

For Windows users, WSL provides a Linux environment directly in Windows:

- **Why use it**: Consistent Unix environment, better compatibility with data science tools
- **Quick install**: In PowerShell (as Admin): `wsl --install`, then restart
- **File access**: Windows files at `/mnt/c/...`, WSL files at `\\wsl$\Ubuntu\...`
- **Best terminal**: Windows Terminal or VS Code's integrated terminal

## Local setup

MacOS:

- [Meet HomeBrew (brew.sh)](https://brew.sh)
- [Data Science Setup on MacOS](https://engineeringfordatascience.com/posts/setting_up_a_macbook_for_data_science/)
- [How I set up my new Macbook Pro for Programming and Data Science](https://towardsdatascience.com/how-i-set-up-my-new-macbook-pro-for-programming-and-data-science-505c69d2142)

Windows:

- [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)
- [A usable and good-looking automation environment on Windows](https://www.trueneutral.eu/2021/win-proper-env.html)

iOS:

_if you're a weirdo and want to turn your iPad into a fully-fledged development environment_

- git: [Working Copy](https://workingcopyapp.com)
- Terminal: [blink.sh](https://blink.sh)
- VS Code: [vscode.dev](https://vscode.dev)
- Jupyter: [Juno](https://juno.sh) (and Juno Connect to use cloud processing and GPUs)

  

Tools you'll need:

- git
    - `brew install git`
    - WSL has git installed by default
    - [GitHub Desktop](https://desktop.github.com) has a GUI (excellent for beginners, but plenty of devs use it, too!)
    - VS Code 👇 can also manage git repositories!
- Python 3 - [Data Science with Python Tutorial](https://www.geeksforgeeks.org/data-science-tutorial/)
    - We'll install later 👇
- R - [R for Data Science](https://r4ds.had.co.nz)
    - [Posit](https://posit.co) (formerly RStudio)
    - [tidyverse](https://www.tidyverse.org/) (has most everything you need)
- Bonus!
    - VS Code - the default IDE for everyone (except people using Posit/RStudio)
        - [Download](https://code.visualstudio.com) or `brew install visual-studio-code`
        - Can also run inside a web browser: [vscode.dev](http://vscode.dev), [Codespaces](https://github.com/features/codespaces)
        - Extensions: Python, Jupyter, GitHub Repositories, Remote Repositories (manage git with VS Code), GitHub Codespaces (cheap remote computer), GitHub Copilot (AI assistant)
    - Fonts (make it nice!)
        - [Fira Mono](https://fonts.google.com/specimen/Fira+Mono?category=Monospace) `brew install font-fira-mono`
        - [Source Code Pro](https://fonts.google.com/specimen/Source+Code+Pro) `brew install font-source-code-pro`
        - Or any other [monospaced font](https://fonts.google.com/?category=Monospace) you like!

### Cloud options

You can run R and python in lots of places, many for free:

- GitHub Codespaces (free extra hours for students with GitHub Education, can work with private repos)
- Google Colab (free for public notebooks, paid for private or higher-powered machines)
- Paperspace (free for public notebooks, paid for private or higher-powered machines)
- Binder (free, always public)

# Markdown

Lightweight markup language for documentation, used in GitHub, Notion, and more:

- **Resources**: [Markdown Guide](https://www.markdownguide.org/basic-syntax/), [Interactive Tutorial](https://www.markdowntutorial.com)

## Key Syntax

- **Paragraphs**: Separate with blank lines
- **Headers**: `# H1`, `## H2`, `### H3`
- **Formatting**: `**bold**`, `_italic_`, `` `code` ``
- **Lists**: 
  - Unordered: `* item` or `- item`
  - Ordered: `1. item` (numbers don't matter)
  - Checklists: `- [ ]` and `- [x]`
- **Code blocks**: Triple backticks ` ``` `
- **Links**: `[text](url)`
- **Blockquotes**: `> quoted text`

Every repo should have a README.md to explain what it is and how to use it.

# LIVE DEMO!

# git and GitHub

Version control system for tracking changes and collaborating on code:

- **Resources**: [Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials/what-is-version-control) (focus on _Getting Started_ and _Collaborating_)

## Essential Git Commands

- **Setup**: `git config --global user.name "Your Name"` and `git config --global user.email "email@example.com"`
- **Starting**: `git init` (new repo) or `git clone URL` (copy existing repo)
- **Basic workflow**:
  1. `git status` (check what's changed)
  2. `git add filename` (stage changes)
  3. `git commit -m "Message"` (save snapshot)
  4. `git push` (upload to remote) / `git pull` (download from remote)

## Collaboration Features

- **Branches**: Create separate workspaces with `git branch` and `git checkout`
- **Pull Requests**: Request code review before merging changes
- **Forks**: Make your own copy of someone else's repository

## Important Notes

- **Never commit sensitive info**: No passwords, PHI, or PII
- **Handling conflicts**: Use `git restore`, `git rebase`, or `git stash` when things get messy
- **GitHub alternatives**: GitLab, Bitbucket, or UCSF's internal GitHub (for PHI)

![](git_branches.png)
![](xkcd_git.png)

# LIVE DEMO!

# Python

The most popular language for data science and machine learning:

- **Resource**: [A Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/) (free online)

## Quick Setup

- **Mac**: `brew install python`
- **Windows**: In WSL: `sudo apt install python3 python3-pip python3-venv`

## Key Packages

- **Data analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine learning**: scikit-learn, PyTorch, TensorFlow/Keras

## Virtual Environments

Isolated Python environments for different projects:

- **Why**: Avoid dependency conflicts between projects
- **How**:
  1. Create: `python3 -m venv env_folder`
  2. Activate: `source env_folder/bin/activate` (Mac/Linux) or `env_folder\Scripts\activate` (Windows)
  3. Install: `pip install -r requirements.txt`
  4. Deactivate: `deactivate`

## Jupyter Notebooks

Interactive Python environment combining code, output, and documentation:

- **Best practice**: Clear outputs before committing to git
- **Why**: Prevents large file sizes and merge conflicts

![Jupyter clearing dialog](jupyter_clear.png)

# LIVE DEMO!

# This Week's Assignment

## GitHub Classroom Overview

- **What**: Platform for distributing, submitting, and grading assignments
- **How**: Accept assignment link → Get private repo → Make changes → Push to submit
- **Benefits**: Automated testing, private repos, direct feedback

## Assignment Tasks

1. **Create README.md** with:
   - Brief introduction (first name only)
   - What you hope to get from the course
   - Music recommendation with link

2. **Write Python script** that:
   - Takes email address as command line argument
   - Hashes it using specified algorithm
   - Outputs to 'hash.email' file

3. **Submit** via git push (auto-graded)

See [exercise.md](exercise.md) for more details and additional practice resources.

# It came from the Internet

Thanks this week to [Data Science Weekly Newsletter](https://datascienceweekly.substack.com/?utm_source=substack&utm_medium=email)

### Data teams

> [!info] Should You Measure the Value of a Data Team?  
> Data teams are sometimes asked to prove their ROI to senior leadership to justify a budget for new hires, tools, projects, or process changes.  
> [https://medium.com/the-prefect-blog/should-you-measure-the-value-of-a-data-team-95c447f28d4a](https://medium.com/the-prefect-blog/should-you-measure-the-value-of-a-data-team-95c447f28d4a)  

> [!info] Data scientists work alone and that's bad | Ethan Rosenthal  
> In Need of a Good Editor Growing up, I had always considered myself a decent writer based on my decent grades in English class.  
> [https://www.ethanrosenthal.com/2023/01/10/data-scientists-alone/](https://www.ethanrosenthal.com/2023/01/10/data-scientists-alone/)  

### Tooling updates

> [!info] Beyond Pandas - working with big(ger) data more efficiently using Polars and Parquet  
> As data scientists/engineers, we often deal with large datasets that can be challenging to work with.  
> [https://medium.com/data-analytics-at-nesta/beyond-pandas-working-with-big-ger-data-more-efficiently-using-polars-and-parquet-fd980353cc2](https://medium.com/data-analytics-at-nesta/beyond-pandas-working-with-big-ger-data-more-efficiently-using-polars-and-parquet-fd980353cc2)  

> [!info] SQL should be your default choice for data engineering pipelines  
> Originally posted: 2023-01-30.  
> [https://www.robinlinacre.com/recommend_sql/](https://www.robinlinacre.com/recommend_sql/)  

### Data science in practice

> [!info] I Used Computer Vision To Destroy My Childhood High Score in a DS Game  
> I train an object detection model to control my computer to play a minigame running in a DS emulator endlessly.  
> [https://betterprogramming.pub/using-computer-vision-to-destroy-my-childhood-high-score-in-a-ds-game-38ebd53a1d64](https://betterprogramming.pub/using-computer-vision-to-destroy-my-childhood-high-score-in-a-ds-game-38ebd53a1d64)  

> [!info] Data Cleaning Plan  #FIXME:MOVE TO NEXT WEEK
> Data cleaning or data wrangling is the process of organizing and transforming raw data into a dataset that can be easily accessed and analyzed.  
> [https://cghlewis.github.io/mpsi-data-training/training_4.html](https://cghlewis.github.io/mpsi-data-training/training_4.html)
t
