# Health Data Science Demos: Tools & Foundations

This document contains three hands-on demos for the first lecture, strategically placed at the ⅓, ⅔, and end points of the lecture. Each demo builds on the previous one, creating a cohesive learning experience.

## Demo 1: Command Line for Health Data Management (15 minutes)

> **SPEAKING NOTES:**
> - **OBJECTIVES:** Familiarize students with terminal navigation, file operations, and basic data manipulation
> - **TIMING:** Present this demo after covering Command Line Basics section
> - **PREPARATION:** Have terminal open and ready to demonstrate
> - **APPROACH:** 
>   - Demonstrate each command, explaining what it does before executing
>   - Emphasize how these commands apply to health data workflows
>   - Point out that these skills form the foundation for data pipeline creation
> - **MISCONCEPTIONS TO ADDRESS:**
>   - "The command line is outdated" - Explain it's still essential for data science
>   - "I might break something" - Reassure that basic commands are safe with proper care
>   - "This is too technical" - Connect to familiar health data tasks
> - **VALIDATION:** Students may show confusion when executing commands or understanding concepts. Common stumbling blocks include remembering command syntax and understanding the purpose of each command.

### Introduction (1 minute)

"Let's explore how the command line can help us organize and analyze health data. These skills are essential for creating reproducible research workflows."

### Part 1: Setting Up a Health Research Project (5 minutes)

1. Open your terminal:
   - Mac: Terminal app
   - Windows: WSL Ubuntu terminal (recommended) or PowerShell
   - Linux: Terminal app

2. Navigate and explore:
   ```bash
   pwd                  # Shows current location (Present Working Directory)
   ls -la               # Lists all files including hidden ones
   cd ~                 # Go to home directory
   mkdir health_project # Create project directory
   cd health_project    # Move into project directory
   ```

3. Create a research project structure:
   ```bash
   # Create organized directory structure for health research
   mkdir -p data/{raw,processed,external} code results figures
   
   # Create a README file
   echo "# Health Data Analysis Project" > README.md
   echo "Project for analyzing patient outcomes data" >> README.md
   
   # View the structure
   ls -la
   find . -type d | sort  # Shows directory tree
   ```

### Part 2: Working with Health Data Files (7 minutes)

1. Create sample health data:
   ```bash
   # Copy the sample data file
   cp /path/to/patients.csv data/raw/
   
   # View the data
   cat data/raw/patients.csv
   ```
   
   Note: The sample data file is available in the demos folder: [patients.csv](patients.csv)

2. Analyze the data with command line tools:
   ```bash
   # Count total records
   wc -l data/raw/patients.csv
   
   # Find patients with hypertension
   grep "hypertension" data/raw/patients.csv
   
   # Count hypertension cases
   grep "hypertension" data/raw/patients.csv | wc -l
   
   # Extract just the blood pressure readings
   cut -d',' -f3 data/raw/patients.csv
   
   # Create a filtered dataset of hypertension patients
   grep "hypertension" data/raw/patients.csv > data/processed/hypertension_patients.csv
   cat data/processed/hypertension_patients.csv
   ```

3. Data pipeline example:
   ```bash
   # A simple data processing pipeline
   cat data/raw/patients.csv | grep -v "patient_id" | cut -d',' -f2,5 | sort > data/processed/age_diagnosis.csv
   
   # View the result
   cat data/processed/age_diagnosis.csv
   ```

### Quick Exercise (2 minutes)

"Now it's your turn! Create a file called `notes.txt` with a few lines about health data topics you're interested in, then use `grep` to find specific terms in it."

```bash
# Example solution
echo "I'm interested in diabetes research" > notes.txt
echo "Also curious about heart disease prevention" >> notes.txt
echo "Want to learn about medical imaging analysis" >> notes.txt
grep "heart" notes.txt
```

### Wrap-up (1 minute)

"These command line skills are the building blocks for more complex data processing. Next, we'll see how to track changes to our work using Git."

## Demo 2: Git & GitHub for Health Research Projects (15 minutes)

> **SPEAKING NOTES:**
> - **OBJECTIVES:** Demonstrate version control basics using VS Code and GitHub interfaces
> - **TIMING:** Present after covering Git and GitHub section
> - **PREPARATION:** 
>   - Have VS Code installed with GitHub extension
>   - Have GitHub account ready
>   - Clear browser cache or use incognito if showing login
> - **APPROACH:**
>   - Focus on visual interfaces rather than command line
>   - Show both VS Code Git tools and GitHub web interface
>   - Emphasize reproducibility aspects for health research
> - **MISCONCEPTIONS TO ADDRESS:**
>   - "Git is only for software developers" - Show research applications
>   - "Version control is too complicated" - Show how GUIs simplify the process
>   - "I work alone, so I don't need Git" - Explain benefits for solo researchers
> - **VALIDATION:** Students may struggle with basic Git concepts like commit, push, and repository structure. Common questions include the difference between local and remote repositories and when to commit changes.

### Introduction (1 minute)

"Version control is essential for reproducible health research. Let's explore how Git and GitHub can help track changes to your analysis code and data processing scripts using visual tools that make the process more intuitive."

### Part 1: Setting Up Git and GitHub (3 minutes)

1. GitHub account setup:
   - Navigate to [GitHub](https://github.com)
   - Show profile settings
   - Highlight GitHub Education benefits for students
   - Discuss privacy considerations for health researchers

2. VS Code Git integration:
   - Show Git extension in VS Code
   - Explain how it simplifies Git operations
   - Point out Source Control icon in the sidebar

### Part 2: Creating a Local Health Research Project (4 minutes)

1. Create a new project folder:
   - Open VS Code
   - Create a new folder: "health-data-analysis"
   - Create a simple README.md file
   
   Note: A sample README.md is available in the demos folder: [README.md](README.md)

2. Initialize Git repository:
   - Click on Source Control icon in VS Code sidebar
   - Click "Initialize Repository" button
   - Show the .git folder that appears (may need to show hidden files)
   - Explain what initialization does

3. Create a simple analysis script:
   - Create a new file: `analyze_outcomes.py`
   
   Note: A sample script is available in the demos folder: [analyze_outcomes.py](analyze_outcomes.py)

### Part 3: First Commit and Publishing to GitHub (4 minutes)

1. Make your first commit:
   - Show files in Source Control panel (they appear with U for Untracked)
   - Stage changes by clicking + icon next to files
   - Enter a commit message: "Initial project setup with risk calculator"
   - Click the checkmark to commit
   - Explain what a commit is and why descriptive messages matter

2. Create and publish to GitHub repository:
   - Click "Publish to GitHub" in Source Control panel
   - Choose public or private repository
   - Name the repository "health-data-analysis"
   - Add a description
   - Select files to include (typically all)
   - Click "Publish"
   - Show the repository on GitHub after publishing

3. Explain GitHub repository features:
   - README.md display on the main page
   - Code browsing and history
   - Issues for tracking tasks and bugs
   - Settings for managing access and integrations

### Part 4: Making Changes and Syncing (3 minutes)

1. Make changes to your code:
   - Add a new function to `analyze_outcomes.py`:
   
   ```python
   def calculate_bmi(weight_kg, height_m):
       """Calculate Body Mass Index"""
       return weight_kg / (height_m ** 2)
   
   # Add example BMI calculation
   if __name__ == "__main__":
       # Existing code remains...
       
       # Add BMI examples
       print("\nBMI Calculations:")
       print(f"BMI for 70kg, 1.75m: {calculate_bmi(70, 1.75):.1f}")
   ```

2. Commit and push changes using VS Code:
   - Show modified files in Source Control panel (M for Modified)
   - Stage changes
   - Enter commit message: "Add BMI calculation function"
   - Commit changes
   - Click "Sync Changes" to push to GitHub
   - Show the updated repository on GitHub

3. Demonstrate viewing history:
   - Show commit history in VS Code (click on "Commits" in Source Control panel)
   - Show history on GitHub (click on "commits" link)
   - Explain how this creates a permanent record of research code evolution

### Quick Exercise (2 minutes)

"Your turn! Create a new file in your repository called `research_question.md` that describes a health data science question you're interested in exploring. Use VS Code to commit and push it to GitHub."

### Wrap-up (1 minute)

"VS Code and GitHub make version control accessible without memorizing complex commands. This approach ensures your health research is reproducible, backed up, and easily shared with collaborators. As you become more comfortable, you can explore additional features like branches for experimental analysis methods."

## Demo 3: Python for Health Data Analysis (15 minutes)

> **SPEAKING NOTES:**
> - **OBJECTIVES:** Introduce Python basics with health data examples and demonstrate different environments
> - **TIMING:** Present after covering Python and Runtime Environments sections
> - **PREPARATION:** 
>   - Have Python installed and ready
>   - Prepare Jupyter notebook environment
>   - Have example health dataset ready
> - **APPROACH:**
>   - Start with basic syntax in terminal
>   - Progress to script-based analysis
>   - Finish with Jupyter notebook for exploratory analysis
>   - Show both local and cloud options
> - **MISCONCEPTIONS TO ADDRESS:**
>   - "Python is too hard to learn" - Show readable syntax
>   - "I need to memorize everything" - Emphasize looking up documentation
>   - "Python isn't suitable for health data" - Show relevant examples
> - **VALIDATION:** Students may struggle with Python syntax, particularly indentation and function definitions. Common questions include how to structure code and when to use different data types.

### Introduction (1 minute)

"Python has become the language of choice for health data science due to its readability and powerful libraries. Let's explore how to use Python for health data analysis in different environments."

### Part 1: Python Basics with Health Data Examples (4 minutes)

1. Start Python in interactive mode:
   ```bash
   python3
   ```

2. Demonstrate basic data types with health examples:
   ```python
   # String (text) - always anonymized for teaching
   patient_name = "Jane Doe"
   
   # Numeric data types
   age = 65                    # Integer
   temperature = 98.6          # Float
   heart_rate = [72, 75, 70]   # List
   
   # Boolean
   has_hypertension = True
   
   # Dictionary for structured data
   patient = {
       "id": "A12345",
       "age": 65,
       "conditions": ["hypertension", "arthritis"],
       "medications": 3
   }
   
   # Accessing data
   print(f"Patient age: {patient['age']}")
   print(f"First condition: {patient['conditions'][0]}")
   
   # Simple calculations
   average_heart_rate = sum(heart_rate) / len(heart_rate)
   print(f"Average heart rate: {average_heart_rate}")
   
   # Control flow
   if temperature > 100.4:
       print("Fever detected")
   else:
       print("Temperature normal")
   
   # Exit interactive mode
   exit()
   ```

### Part 2: Creating a Health Data Analysis Script (4 minutes)

1. Demonstrate a script for analyzing patient vitals:
   
   Note: The script is available in the demos folder: [vitals_analysis.py](vitals_analysis.py)
   
   ```bash
   # Run the script
   python3 vitals_analysis.py
   ```

2. Explain key Python concepts:
   - Functions and docstrings
   - Data structures (lists, dictionaries)
   - Control flow (if/else, loops)
   - String formatting
   - Basic calculations

### Part 3: Jupyter Notebooks for Exploratory Health Data Analysis (4 minutes)

1. Launch Jupyter Notebook (or show a prepared notebook):
   
   Note: A sample notebook is available in the demos folder: [health_data_exploration.ipynb](health_data_exploration.ipynb)
   
   ```bash
   # In a real demo, you would have Jupyter installed
   jupyter notebook health_data_exploration.ipynb
   ```

2. Explain Jupyter Notebook features:
   - Mixing code, output, and documentation
   - Interactive execution
   - Rich output (tables, charts)
   - Ideal for exploratory analysis and sharing research

### Part 4: Cloud Options for Python in Health Research (3 minutes)

1. Demonstrate GitHub Codespaces (or describe with screenshots):
   - Show how to create a Codespace from a repository
   - Highlight VS Code in browser features
   - Discuss benefits for collaboration

2. Introduce Google Colab (or describe with screenshots):
   - Navigate to [Google Colab](https://colab.research.google.com)
   - Show how to create a new notebook
   - Demonstrate GitHub integration
   - Emphasize NEVER using PHI data in Colab

3. Discuss when to use each environment:
   - Local Python: For sensitive data, full control
   - Jupyter Notebooks: For exploratory analysis, sharing results
   - Codespaces: For collaboration, consistent environments
   - Colab: For quick experiments, GPU access (no PHI!)

### Quick Exercise (2 minutes)

"Your turn! Modify the vitals analysis script to add a new function that calculates BMI given height and weight, then run it to see the results."

```python
# Example solution to add to vitals_analysis.py
def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index"""
    return weight_kg / (height_m ** 2)

# Test the function
print("\nBMI Calculations:")
print(f"BMI for 70kg, 1.75m: {calculate_bmi(70, 1.75):.1f}")
```

### Wrap-up (1 minute)

"Python provides powerful tools for health data analysis. As you progress, you'll learn more about specialized libraries like Pandas, NumPy, and scikit-learn that make complex analysis easier. Remember to always consider data security when working with health information."

## Final Thoughts

These demos are designed to build on each other, creating a cohesive learning experience:

1. **Command Line Demo**: Establishes basic skills for file management and simple data operations
2. **Git & GitHub Demo**: Builds on file management to introduce version control and collaboration
3. **Python Demo**: Leverages the previous skills to perform actual data analysis

Each demo includes health-specific examples to make the content relevant and engaging for health data science students.
