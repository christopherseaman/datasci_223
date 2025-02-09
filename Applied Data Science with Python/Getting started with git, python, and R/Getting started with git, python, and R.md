It doesn’t matter which tools you use; python and R (and other specialized tools) are quite capable. Since python and R are the most commonly used tools, knowing one or both of them will make it easier to play well with others. Don’t try to be an expert in everything! Figure out which you prefer and learn to be “fluent” (able to code a solution from start to finish) in one, then you can get by being “conversational” (able to read and edit others’ code) in the other.

Additionally, collaboration usually happens in git and documentation will use markdown. Luckily, those are easy to pick up. _Almost as easy as SQL._

# `git init`

- Tools: `python`, `R`, and `git`
    - Getting set up locally
    - Cloud options (Colab, Binder, Paperspace, GitHub)
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

# Installing tools

For most roles, data science happens in `python` and `R`

## Quickstart

_Note: this is also included in the week’s assignment_

These are the standard options that I’ll be using to demonstrate going forward. They will also give us a common base to work from, so we can focus on the work rather than tweaking/fixing our development environment.

- Sign up for an account on [GitHub](https://github.com)
- Install Python 3 ([instructions](https://docs.python-guide.org))
- Get [VisualStudio Code](https://code.visualstudio.com)
    
    - Most commands are accessed using the “Command Palette”
        - **Shift + Command + P** (Mac)
        - **Ctrl + Shift + P** (Windows/Linux)
    
    - Extensions
        - Python + Jupyter (use notebooks within VS Code)
        - GitHub Repositories + Remote Repositories (manage git in VS Code instead of the terminal)

**Note:** If you do not have space to install software, it is possible to follow along using [Google Colab](http://colab.research.google.com) but _never_ use PHI data with public-facing tools.

## Additional options

### Local setup

MacOS:

- [Meet HomeBrew (brew.sh)](https://brew.sh)
- [Data Science Setup on MacOS](https://engineeringfordatascience.com/posts/setting_up_a_macbook_for_data_science/)
- [How I set up my new Macbook Pro for Programming and Data Science](https://towardsdatascience.com/how-i-set-up-my-new-macbook-pro-for-programming-and-data-science-505c69d2142)

Windows:

- [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install)
- [A usable and good-looking automation environment on Windows](https://www.trueneutral.eu/2021/win-proper-env.html)

iOS:

_if you’re a weirdo and want to turn your iPad into a fully-fledged development environment_

- git: [Working Copy](https://workingcopyapp.com)
- VS Code: [blink.sh](https://blink.sh)
- Jupyter: [Juno](https://juno.sh) (and Juno Connect to use cloud processing and GPUs)

  

Tools you’ll need:

- git
    - `brew install git`
    - WSL has git installed by default
    - [GitHub Desktop](https://desktop.github.com) has a GUI (excellent for beginners, but plenty of devs use it, too!)
    - VS Code 👇 can also manage git repositories!
- Python 3 - [Data Science with Python Tutorial](https://www.geeksforgeeks.org/data-science-tutorial/)
    - We’ll install later 👇
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

- GitHub Codespaces (free up to some number of hours/month, can work with private repos)
- Paperspace (free for public notebooks, paid for private or higher-powered machines)
- Google Colab (free for public notebooks, paid for private or higher-powered machines)
- Binder (free, always public)

# Markdown

Markdown is a lightweight markup language for writing documents. The format was created as an alternative to HTML, while retaining most of the capabilities. It’s the most common format in many tools, including GitHub, Notion, and Google Docs (when enabled).

- [https://www.markdownguide.org/basic-syntax/](https://www.markdownguide.org/basic-syntax/) (cheat sheet)
- [https://www.markdowntutorial.com](https://www.markdowntutorial.com/) (self-guided tutorial)

## Document flow

### Paragraphs

Start a new paragraph by separating it from the previous one with a blank line

```JSON
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 

This is a new paragraph!
```

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

This is a new paragraph!

### Headers

Starting a line with hash symbols `#` will creating headings within your document:

- # Header 1 (biggest)
- ## Header 2
- ### Header 3 (smallest commonly supported)

### Font Styles

There is some variability in how these are applied between Slack, Notion, GitHub, etc.

- **Bold** - double-asterisks `**` around a word to ****bold**** (some apps allow single astersisks)
- _italic_ - single underscores `_` around a word to __italicize__ (some apps also confusingly also use single asterisks for italic, too)

> Blockquote - prefix text with a greater-than sign to `>` blockquote (Notion uses the pipe | symbol). To blockquote multiple paragraphs, include a ‘>’ on the blank line between them

```JSON
> This is Blockquoted
```

### Lists

**Unordered lists** start with an asterisks, hyphen, or plus sign. Indent with additional spaces to make sublists:

```JSON
* one
  * two
* three
```

- one
    - two
- three

**Ordered lists** start with numbers and indent similarly to undordered lists, but it actually doesn’t matter which digits you use.

```JSON
1. asdf
  3. jfjf
7. btbk
```

1. asdf
    1. jfjf
2. btbk

**Checklists** start with bracket pairs `[ ]`, completed items with an x inside `[x]`

```JSON
[] do this
[x] this is done
```

- [ ] do this
- [x] this is done

### Code

Code is marked with surrounding backticks ` and `` `can be included inside a paragraph` ``

Larger blocks of code, spanning multiple lines begin and end with three backticks

````JSON
```
This is a large block of code

across multiple lines
```
````

```JSON
This is a large block of code

across multiple lines
```

### Links

You can create links by surrounding the link text with [] brackets, then the url surrounded by parentheses:

```JSON
Neat collection of [data science notes](badmath.org/datasci)
```

Neat collection of data science notes

# git

Atlassian has an [excellent tutorial on git](https://www.atlassian.com/git/tutorials/what-is-version-control), the _Getting Started_ and _Collaborating_ sections will be most applicable early on.

## Local setup

You won’t be using the terminal for everything, but it is incredibly useful to be familiar with it as a tool.

### Configuring with your name and email

We need to tell git who we are. We do this using `git config` to

`git config --global user.name "<YOUR NAME>"`

`git config --global user.email "<YOUR EMAIL>"` to set your email address

> [!important]  
> Note! Having your email address listed in a public repository is a bad idea. You will get targeted for spam or worse. GitHub will set up an anonymous proxy email address automatically, you can find it https://github.com/settings/emails while logged in.  

![[github_email.png]]

## Cloud options

We’ll work with GitHub here, but other options include [GitLab](https://gitlab.com/) and [Bitbucket](https://bitbucket.org/). UCSF also has an [internally-facing version of GitHub](https://it.ucsf.edu/search?search=github), which you should definitely use if you’re working on anything PHI-related. Access to UCSF’s GitHub and High-Performance Computing (Wynton) must be requested from IT.

## Git init & clone

You can create a new repository in the current directory with the `git init` command. This adds a hidden folder `.git` with the configuration for the repo. You can later add the repo to a remote host (like GitHub) where others can access it

To copy a remote repository to your local machine use the `git clone` command. This will copy the repository and all its version history to a subdirectory with the same name as the repository you’ve cloned.

To clone the notebooks that accompany the book Python for Data Analysis:

`git clone` `[https://github.com/wesm/pydata-book.git](https://github.com/wesm/pydata-book.git)`

![[git_clone.png]]

## Commit

You save a snapshot of your work using `git commit` commands. Each commit will need a short message to describe the changes you’ve made.

1. `git status` - to see what changes you’ve made since the last commit
2. `git add <FILE>` - add files you’ve changed to _staging_ (included in the next commit)
3. `git commit -m <MESSAGE>` - commit your work with a quick summary

  

## Push ⇄ pull

Does what it says on the tin.

- `git push` will add your local commits to the remote copy
- `git pull` will download any changes from the remote to your local copy
- `git sync` (not all systems) will perform both a pull and push

## Fork, branch, and merge

Sometimes you want to work on something outside the “main” flow of a repository. Maybe there’s an analysis or model you’re working on that isn’t complete. By creating a separate **branch** of the repository, you can do your work without worrying about breaking the **main** or **trunk** branch of the repository. If you want to make your own work based on another repository, you can **fork** it, creating a copy that you own going forward.

### Git branch workflow

It makes it a lot easier when collaborating with others to keep a clean and functional `main` branch. If you’re an imperfect human, then you probably can’t ensure that every commit you make along the way is also clean and functional. Mistakes happen.

One solution is to use a **branch workflow**, where work-in-progress happens in dedicated branches. Once a piece of work is deemed complete, it can be merged into the main branch.

A best practice when merging work into the main branch is to use a **pull request** or **PR**. A pull request signals that your work may be complete and you’d like someone else to review it and give feedback. This ensures not just that the changes you’ve made are correct, but that they are understandable to others. Once the reviewer gives the 👍 (and conflicts are resolved), your development branch can be merged into the main branch.

![[Notion/Getting into Data Science/Applied Data Science with Python/Getting started with git, python, and R/git_branches.png|git_branches.png]]

## Sensitive information

Never ever. Not once. This goes for passwords, PII, and PHI. Don’t put it on GitHub.

## Conflict resolution

Sometimes your repos will get into states that can’t be resolved automatically and non-destructively. There are a few ways to resolve git conflicts:

- `git restore <FILE>` - discard all changes to <FILE> since the last commit
- `git rebase` - discard all changes
- `git stash` - save the current state in a local “stash”, then rebase the repo to the last commit

  

Read more on merge conflicts in [Atlassian’s tutorial](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)

![[xkcd_git.png]]

  

# Python

We’ll just barely scratch the surface here, it goes a **lot** deeper. This section will follow the same format as _A Whirlwind Tour of Python_, which goes quite a bit deeper while staying under a hundred pages.

- A Whirlwind Tour of Python ([free on the web](https://jakevdp.github.io/WhirlwindTourOfPython/), [github](https://github.com/jakevdp/WhirlwindTourOfPython), [oreilly](https://www.oreilly.com/library/view/a-whirlwind-tour/9781492037859/))

## Installing

- Package manager
    - MacOS: [Homebrew](https://brew.sh)
    - Windows: [Chocolatey](https://chocolatey.org/install)
- Python 3
    - Mac: `brew install python`
    - Windows: `choco install python`

- FYI - You’ll see these around but don’t need to install right now
    - Common packages:
        - [Jupyter](https://jupyter.org)
        - [Pandas](https://pandas.pydata.org)
        - [Numpy](https://numpy.org)
        - [scikit-learn](https://scikit-learn.org/)
    - ML frameworks:
        - [PyTorch](https://pytorch.org) (high-level: [Lightning](https://lightning.ai), [fast.ai](https://docs.fast.ai))
        - [TensorFlow](https://www.tensorflow.org) (high-level: [Keras](https://keras.io))

## Hands-on in the cloud with Colab

1. (bonus points) On GitHub, fork [the Whirlwind Tour repo](https://github.com/jakevdp/WhirlwindTourOfPython) by clicking the Fork button near the top of the page.
    1. This will create your own copy of the repo, where you can commit changes
    2. Keep this tab open, you'll need it in a few steps.
2. Go to https://colab.research.google.com and sign in to your google account
3. A modal should pop up with multiple tabs, select GitHub
    1. You can open this modal at any time using File > Open notebook
4. Enter the URL of your forked copy of the Whirlwind Tour repo and hit Enter
5. A list of notebooks should populate below, one for each chapter
6. Choose a chapter's notebook and open it by clicking on its name; e.g., `02-Basic-Python-Syntax.ipynb`
7. The notebook should open in a browser window
8. Select and edit cells however you want
9. Run cells by clicking the "play" icon at top left of it, or by hitting shift+enter while the cell is selected
    1. A scary warning will pop up the first time you run a cell. You can dismiss it by clicking "Run anyway".
    2. It is possible that a malicious notebook could request access to your data, but this one should be safe.
10. (bonus points) Commit changes you've made to GitHub directly from Colab. The “Cannot save changes” warning is misleading. To commit changes:
    1. Click File > Save a copy in GitHub (you may need to sign in to GitHub)
    2. This should open a dialogue with the repo, branch, and file name of the notebook you're working in
    3. Edit the commit message and click "OK"

## Virtual Environments

Just a quick note on environments. Sometimes you’ll be working with projects that have different requirements. Perhaps you need to work on a repository that hasn’t been touched in awhile and it only works with older versions of a library.

To resolve these kinds of conflicts we use:

- **requirements.txt** - a text file listing the libraries and versions expected by the repository
- **virtual environments** - a simulated python environment with its own copies of python and libraries separate from the system

Virtual environments work by:

1. `python3 -m venv <FOLDER>` - create an environment (usually within the repo) inside a particular folder
2. `source <FOLDER>/bin/activate` - activate the virtual environment
3. `pip install -r requirements.txt` - install the libraries/versions listed
4. `deactivate` - exit the virtual environment and return to the system environment

## Jupyter notebooks and git

Notebooks capture a lot of feedback and output when run, which can get quite large. To keep your repository light, remember to **Clear All Outputs** before committing changes to a notebook.

![[jupyter_clear.png]]

# Exercise

- Setup: Get everything installed from the Quickstart
    - (Recommended) Package manager
        - MacOS: [Homebrew](https://brew.sh)
        - Windows: [Chocolatey](https://chocolatey.org/install)
    - Python 3
    - VS Code
- Sign up for a GitHub account if you haven’t already
- Git Basics: Read through the _Getting Started_ and _Collaborating_ sections of the [Atlassian git tutorial](https://www.atlassian.com/git/tutorials/setting-up-a-repository), really (okay to visit this first if you need help creating a fork of the Whirlwind Tour repository)
    - Microsoft also has a write-up on [Getting started with GitHub and Visual Studio Code](https://learn.microsoft.com/en-us/training/paths/get-started-github-and-visual-studio-code/)
- (Recommended) Python Foundations: Create an account on [Exercism](http://exercism.org/tracks/python) and work through the [python track](http://exercism.org/tracks/python) until you get to loops. That exercise is called _Making the Grade_, which you will complete and submit via GitHub.
    - Python Track on Exercism
        
        ![[exercism_python.png]]
        
- Coding assignment:
    - Fork the class repo on GitHub - [https://github.com/christopherseaman/datasci_223](https://github.com/christopherseaman/datasci_223)
    - Complete the _Making the Grade_ exercise: `exercises/1-foundations/exercise.ipynb`
    - Create a `Readme.md` in the root directory of the repo and introduce yourself (or your team), create a link to your solution to _Making the Grade_, and any notes on what you’re hoping to get out of the course or difficulties you had with the assignment
    - Commit your changes to your forked copy of the repository on GitHub
    - Submit the assignment by letting us know where to find the repository
- (Optional) Work on the [Whirlwind Tour of Python](https://jakevdp.github.io/WhirlwindTourOfPython/) chapters (through ch7, “Control Flow Statements”) alongside the notebooks [on GitHub](https://github.com/jakevdp/WhirlwindTourOfPython)
    1. Work through examples in your own development branch
    2. Commit your changes when you’ve reached good stopping points
    3. When you’re ready for feedback, share it with someone else
    4. Try more [exercises on python basics](https://pythonbasics.org/exercises/) or search the web for exercises/examples specific to what you want to practice. [Python is the second-most popular programming language](https://octoverse.github.com/2022/top-programming-languages) (javascript runs the web at \#1), so there’s a lot of material out there.
- (Optional, prep for next week) Finished everything? Start [Python for Data Analysis](https://wesmckinney.com/book/) to prepare for data munging next lecture. The [book has a repo on GitHub](https://github.com/wesm/pydata-book/tree/3rd-edition) for the examples and exercises

# It came from the Internet

Thanks this week to [Data Science Weekly Newsletter](https://datascienceweekly.substack.com/?utm_source=substack&utm_medium=email)

### Data teams

> [!info] Should You Measure the Value of a Data Team?  
> Data teams are sometimes asked to prove their ROI to senior leadership to justify a budget for new hires, tools, projects, or process changes.  
> [https://medium.com/the-prefect-blog/should-you-measure-the-value-of-a-data-team-95c447f28d4a](https://medium.com/the-prefect-blog/should-you-measure-the-value-of-a-data-team-95c447f28d4a)  

> [!info] Data scientists work alone and that's bad | Ethan Rosenthal  
> In Need of a Good Editr Growing up, I had always considered myself a decent writer based on my decent grades in English class.  
> [https://www.ethanrosenthal.com/2023/01/10/data-scientists-alone/](https://www.ethanrosenthal.com/2023/01/10/data-scientists-alone/)  

### Tooling updates

> [!info] JupySQL: Better SQL in Jupyter  
> Eduardo Blancas TL;DR; we forked ipython-sql ( pip install jupysql) and are actively developing it to bring a modern SQL experience to Jupyter!  
> [https://ploomber.io/blog/jupysql/](https://ploomber.io/blog/jupysql/)  

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

> [!info] Data Cleaning Plan  
> Data cleaning or data wrangling is the process of organizing and transforming raw data into a dataset that can be easily accessed and analyzed.  
> [https://cghlewis.github.io/mpsi-data-training/training_4.html](https://cghlewis.github.io/mpsi-data-training/training_4.html)