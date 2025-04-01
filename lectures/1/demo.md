# Demo 1: Command Line Basics

Let's explore the command line together:

1. Opening the terminal
   - Mac: Terminal app
   - Windows: WSL Ubuntu terminal
   - Linux: Terminal app

2. Basic navigation
   - `pwd` - Where am I?
   - `ls` - What's here?
   - `cd` - Moving around
   - Special directories: `~`, `.`, `..`

3. Creating a project structure
   - `mkdir project`
   - `cd project`
   - `mkdir data code results`
   - `touch README.md`
   - `ls -la`

4. Viewing and editing files
   - `echo "# My Project" > README.md`
   - `cat README.md`
   - `nano README.md` (add a description)
   - `cat README.md`

5. Chaining commands
   - `ls | grep "README"`
   - `echo "# Files" >> README.md && cat README.md`

# Demo 2: Git and GitHub with Codespaces

Let's explore Git and GitHub Codespaces:

1. GitHub account setup
   - Creating an account
   - Setting up GitHub Education benefits
   - Configuring your profile

2. Creating a repository
   - New repository on GitHub
   - Adding a README.md
   - Adding a .gitignore for Python

3. Using GitHub Codespaces
   - Launching a Codespace from your repository
   - Exploring the VS Code interface in the browser
   - Installing extensions

4. Making changes with Git
   - Creating a new file
   - Staging changes with git add
   - Committing changes with a message
   - Pushing changes to GitHub

5. Branching and merging
   - Creating a new branch
   - Making changes on the branch
   - Creating a pull request
   - Merging the changes

# Demo 3: Python Basics

Let's explore Python in different environments:

1. Running Python
   - In the terminal: `python3`
   - In VS Code/Codespaces
   - In Jupyter notebooks

2. Basic Python syntax
   - Variables and data types
   - Simple operations
   - Control flow (if/else, loops)
   - Functions

3. Creating a simple script
   - Writing a Python script
   - Running it from the command line
   - Passing arguments to the script

4. Virtual environments
   - Creating a virtual environment
   - Installing packages
   - Using requirements.txt

5. Connecting to the assignment
   - Overview of the assignment requirements
   - How to submit via GitHub Classroom

# Demo 4: Hands-on in the cloud with Colab

1. On GitHub, fork [the Whirlwind Tour repo](https://github.com/jakevdp/WhirlwindTourOfPython) by clicking the Fork button near the top of the page.
   - This will create your own copy of the repo, where you can commit changes
   - Keep this tab open, you'll need it in a few steps.

2. Go to https://colab.research.google.com and sign in to your google account

3. A modal should pop up with multiple tabs, select GitHub
   - You can open this modal at any time using File > Open notebook

4. Enter the URL of your forked copy of the Whirlwind Tour repo and hit Enter

5. A list of notebooks should populate below, one for each chapter

6. Choose a chapter's notebook and open it by clicking on its name; e.g., `02-Basic-Python-Syntax.ipynb`

7. The notebook should open in a browser window

8. Select and edit cells however you want

9. Run cells by clicking the "play" icon at top left of it, or by hitting shift+enter while the cell is selected
   - A scary warning will pop up the first time you run a cell. You can dismiss it by clicking "Run anyway".
   - It is possible that a malicious notebook could request access to your data, but this one should be safe.

10. Commit changes you've made to GitHub directly from Colab. The "Cannot save changes" warning is misleading. To commit changes:
    - Click File > Save a copy in GitHub (you may need to sign in to GitHub)
    - This should open a dialogue with the repo, branch, and file name of the notebook you're working in
    - Edit the commit message and click "OK"
