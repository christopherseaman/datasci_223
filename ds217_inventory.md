# Python and Command Line Fundamentals - Comprehensive Course Inventory

## Additional Reference Materials
### Command Line Resources
- [LinuxCommand.org](http://linuxcommand.org/lc3_learning_the_shell.php) - Learning the shell
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php) - Free book by William Shotts
- [The Missing Semester](https://missing.csail.mit.edu/) - MIT course on developer tools
- [Bash manual](https://www.gnu.org/software/bash/manual/) - Official Bash documentation
- [PowerShell documentation](https://docs.microsoft.com/en-us/powershell/) - Microsoft's PowerShell docs

### Python Resources
- [Official Python documentation](https://docs.python.org/3/) - Python language reference
- [Codecademy Python course](https://www.codecademy.com/learn/learn-python-3) - Interactive Python learning
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) - Practical Python programming
- [Introduction to Python](http://introtopython.org/) - Python basics tutorial
- [A Byte of Python](https://python.swaroopch.com/) - Python book for beginners

### Git and Markdown Resources
- [Atlassian Git tutorial](https://www.atlassian.com/git/tutorials/what-is-version-control) - Comprehensive Git guide
- [Markdown Guide](https://www.markdownguide.org/basic-syntax/) - Markdown syntax reference
- [Markdown Tutorial](https://www.markdowntutorial.com/) - Interactive Markdown learning
- [CommonMark tutorial](https://commonmark.org/help/tutorial) - CommonMark standard tutorial

### Development Tools
- [Jetbrains' Learn PyCharm](https://www.jetbrains.com/pycharm/learn/) - PyCharm IDE tutorials
- [Effective PyCharm Course](https://training.talkpython.fm/courses/explore_pycharm/mastering-pycharm-ide) - Advanced PyCharm training
- [regex101.com](https://regex101.com/) - Regular expression testing tool

## Course Overview
### Syllabus (DataSci 217)
#### Course Structure
- Learning objectives
  - Develop Python programming proficiency
  - Master command line operations
  - Learn data manipulation with Python libraries
  - Understand version control with Git
- Class structure
  - Lectures with hands-on demos
  - Practical assignments
  - Lab sessions for assignment help

## Lecture 1: Introduction to Python and Command Line Basics
### Command Line Fundamentals
`datasci_217/01/index.md`
#### Key Topics
- Shell environments
  - Different shell types (sh, bash, csh, zsh, PowerShell)
  - Getting to the command line on different operating systems
- File system navigation
  - `pwd`, `ls`, `cd` commands
  - Special directories (`~`, `.`, `..`)
- Basic file operations
  - Creating directories with `mkdir`
  - Creating files with `touch`
  - Copying with `cp`
  - Moving/renaming with `mv`
  - Removing with `rm`
- Viewing file contents
  - `cat`, `head`, `tail`
- Text manipulation
  - Searching with `grep`
  - Chaining commands with pipe `|`

### Python Basics
#### Key Topics
- Python installation and setup
- Running Python
  - Interactive mode (REPL)
  - Script execution
- Syntax fundamentals
  - Indentation
  - Comments
- Basic data types
  - Integers
  - Floats
  - Strings
  - Variables and assignment
- Simple operations
  - Arithmetic operators
  - String concatenation
- Control structures
  - Comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`)
  - Logical operators (`and`, `or`, `not`)
  - Conditional statements (if, elif, else)
  - For loops with `range()`
- String formatting
  - String concatenation
  - f-strings

## Lecture 2: Version Control, Markdown, and Python Environments
### Git and GitHub
`datasci_217/02/index.md`
#### Key Topics
- Git configuration
  - Setting up user name and email
- Repository operations
  - Creating repositories with `git init`
  - Cloning with `git clone`
  - Tracking changes with `git status`
  - Staging with `git add`
  - Committing with `git commit`
- Remote repository interaction
  - Pushing with `git push`
  - Pulling with `git pull`
- Branching and merging
  - Creating branches
  - Pull requests
  - Merge conflict resolution
- Best practices
  - Handling sensitive information
  - Communication with team members
### Advanced Command Line
`datasci_217/03/index.md`
#### Key Topics
- Symbolic links with `ln -s`
- Environment variables
  - Setting with `export`
  - Viewing with `env`
  - Using `.env` files for secrets
- Shell scripts
  - Creating executable scripts
  - Shebang (`#!/bin/bash`)
  - Making scripts executable with `chmod`
  - Passing arguments to scripts
- Scheduled tasks with `cron`
  - Crontab syntax
  - Scheduling recurring tasks
  - Logging cron job output
- File compression and decompression
  - `tar`, `tar.gz`, `tgz`
  - `zip` and `unzip`

### Python Data Structures
#### Key Topics
- Lists
  - Creation and initialization
  - Indexing and slicing
  - Common operations (append, extend, remove)
  - Length with `len()`
  - Generating lists with `range()`
- Strings as sequences
  - Character access
  - String slicing
- Dictionaries
  - Key-value pairs
  - Creation and initialization
  - Adding, accessing, and removing elements
  - Dictionary methods (keys, values, items)
  - Checking for key existence
- Sets
  - Unique elements
  - Creation and initialization
  - Set operations (add, remove, union, intersection, difference)
- Tuples
  - Immutable collections
  - Creation and initialization
  - Tuple unpacking
- Nested data structures
  - Combining lists, dictionaries, and other structures
- Sorting
  - `sort()` vs `sorted()`
  - Sorting dictionaries
- List comprehensions (introduction)

## Lecture 4: File Operations, Functions, and Remote Access
### Python File Operations
`datasci_217/04/index.md`
#### Key Topics
- File handling
  - Opening files with `open()`
  - File modes (read, write, append, binary)
  - Reading with `read()`, `readline()`, `readlines()`
  - Writing with `write()`, `writelines()`
  - Using `with` statement for automatic closing
- Reading files line-by-line
- Splitting lines into arrays with `split()`
- Common file operations
  - Checking if a file exists
  - Deleting files
  - Renaming files
- Printing to files
- Directory operations
  - Creating directories
  - Listing directory contents
  - Checking if a path is a directory

### Python Functions and Modules
#### Key Topics
- Function definition and calling
- Function parameters
  - Default parameters
  - Positional arguments
  - Keyword arguments
  - Variable arguments with `*args` and `**kwargs`
- Command line arguments
  - Using `sys.argv`
  - Using `argparse`
- Modules
  - Importing modules
  - Creating custom modules
  - `if __name__ == "__main__"` pattern

### Remote Access and High-Performance Computing
#### Key Topics
- Jupyter Notebooks
  - Creating and using notebooks
  - Remote Jupyter setup
- SSH (Secure Shell)
  - Basic SSH usage
  - Remote server options (UCSF Wynton, SDF, Google Cloud, GitHub Codespaces)
- File transfer with SCP
- Persistent sessions
  - `screen` usage and commands
  - `tmux` usage and commands
  - `mosh` for mobile connections
- Brief introduction to HPC
  - Wynton HPC basics
  - Job submission with SGE
- GPU computing with Python
  - CUDA introduction
  - Python libraries for GPU computing

## Lecture 5: Data Management with NumPy, Pandas, and Shell Tools
### NumPy Fundamentals
`datasci_217/05/index.md`
#### Key Topics
- NumPy introduction
  - Purpose and capabilities
  - The `ndarray` object
- Creating and manipulating arrays
  - Array creation methods
  - Reshaping arrays
  - Flattening with `flatten()` and `ravel()`
  - Stacking arrays
- Array operations
  - Element-wise operations
  - Matrix operations
  - Universal functions (ufuncs)
- Array attributes and methods
  - Shape, dimensions, size, data type
  - Indexing and slicing
  - Broadcasting

### Pandas for Data Analysis
#### Key Topics
- Pandas introduction
  - Series and DataFrame objects
  - Creating Series and DataFrames
- File I/O
  - Reading data files (CSV, JSON, Excel)
  - Writing data files
- Data cleaning
  - Handling missing data
  - Checking for nulls
  - Dropping and filling missing values
- DataFrame operations
  - Viewing data with `head()` and `tail()`
  - Getting information with `info()` and `describe()`
  - Selecting columns
- Data access methods
  - Label-based indexing with `loc`
  - Integer-based indexing with `iloc`
  - Boolean indexing
- Data analysis
  - Basic statistics
  - Value counts
  - Correlation
  - Grouping and aggregation
- NumPy and Pandas integration
  - Converting between NumPy arrays and Pandas objects
  - Using NumPy functions with Pandas

### Data Munging in the Shell
#### Key Topics
- Text processing with Unix tools
- Extracting columns with `cut`
  - Field selection
  - Custom delimiters
- Character transformation with `tr`
  - Character replacement
  - Character deletion
  - Squeezing repeated characters
- Stream editing with `sed`
  - Search and replace
  - Deleting lines
  - Inserting text
- Regular expressions
  - Basic regex patterns
  - Using regex with grep and sed
- Combining shell commands
  - Building data processing pipelines with pipes
