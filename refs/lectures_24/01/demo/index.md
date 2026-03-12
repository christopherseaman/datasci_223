# Health Data Science Demos: Tools & Foundations

This document contains three hands-on demos for the first lecture, strategically placed at the ⅓, ⅔, and end points of the lecture. Each demo builds on the previous one, creating a cohesive learning experience.

## Demo 1: Command Line for Health Data Management (10 minutes)

### Introduction (1 minute)

> Let's explore how the command line can help us organize and analyze health data. These skills are essential for creating reproducible research workflows.

### Part 1: Setting Up a Health Research Project (3 minutes)

1. Open terminal and navigate:
    ```bash
    pwd                  # Shows current location
    ls -la               # Lists all files including hidden ones
    cd ~                 # Go to home directory
    mkdir health_project # Create project directory
    cd health_project    # Move into project directory
    ```

2. Create research project structure:
    ```bash
    # Create organized directory structure
    mkdir -p data/{raw,processed} code results figures
    
    # View the structure
    ls -la
    find . -type d | sort  # Shows directory tree
    ```

### Part 2: Working with Health Data Files (4 minutes)

1. Copy and view sample health data:
    ```console
    # Copy the sample data file (adjust path as needed)
    cp ../lectures/01/demo/files/patients.csv data/raw/

    # View the data
    cat data/raw/patients.csv
    ```

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
    
    # Create a filtered dataset
    grep "hypertension" data/raw/patients.csv > data/processed/hypertension_patients.csv
    cat data/processed/hypertension_patients.csv
    ```

3. Create a data processing pipeline:
    ```bash
    # A simple data processing pipeline
    echo "age,medication_count" > data/processed/age_diagnosis.csv
    cat data/raw/patients.csv | grep -v "patient_id" | cut -d',' -f2,5 | sort >> data/processed/age_diagnosis.csv
    
    # View the result
    cat data/processed/age_diagnosis.csv
    ```

### Error Handling Examples

If you encounter errors:
1. File not found:
    ```bash
    # Check if file exists
    ls -la data/raw/
    
    # Verify file path
    pwd
    ```

2. Permission denied:
    ```bash
    # Check permissions
    ls -la data/raw/patients.csv
    
    # Fix permissions if needed
    chmod +r data/raw/patients.csv
    ```

### Quick Exercise (2 minutes)

> Create a file called `notes.txt` with a few lines about health data topics you're interested in, then use `grep` to find specific terms in it.

```bash
# Example solution
echo "I'm interested in diabetes research" > notes.txt
echo "Also curious about heart disease prevention" >> notes.txt
echo "Want to learn about medical imaging analysis" >> notes.txt
grep "heart" notes.txt
```

## Demo 2: Git & GitHub for Health Research Projects (10 minutes)

### Introduction (1 minute)

> Version control is essential for reproducible health research. Let's explore two common Git workflows: starting a new project and working with an existing repository.

### Part 1: Starting a New Project (3 minutes)

#### Option A: Create and Publish from VS Code
1. Initialize Git in VS Code:
    - Open VS Code in the project directory
    - Click the Source Control icon in the sidebar (or press Ctrl+Shift+G)
    - Click "Initialize Repository"
    - Stage the patients.csv file by clicking the + icon next to it
    - Enter commit message: "Add sample patient data"
    - Click the checkmark to commit

2. Create GitHub repository:
    - Go to GitHub.com
    - Click the + icon in top right
    - Select "New repository"
    - Name: "health-data-demo"
    - Description: "Demo repository for health data analysis"
    - Choose "Public"
    - Don't initialize with README
    - Click "Create repository"

3. Connect and push:
    - In VS Code, click "Publish to GitHub" in the Source Control panel
    - Choose the repository you just created
    - Click "Publish"
    - Verify the files appear on GitHub

#### Option B: Create on GitHub and Clone Locally
1. Create repository on GitHub:
    - Go to GitHub.com
    - Click the + icon in top right
    - Select "New repository"
    - Name: "health-data-demo"
    - Description: "Demo repository for health data analysis"
    - Choose "Public"
    - Initialize with a README.md
    - Add .gitignore for Python
    - Click "Create repository"

2. Clone to local machine:
    - On GitHub, click the green "Code" button
    - Copy the HTTPS URL
    - Open VS Code
    - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
    - Type "Git: Clone"
    - Paste the repository URL (or if you just start typing it will search for it)
    - Choose where to save it
    - Click "Clone"

3. Make and push changes:
    - Open the cloned repository in VS Code
    - Add your files (e.g., patients.csv)
    - Stage changes in Source Control
    - Commit with a descriptive message
    - Click "Sync Changes" to push

### Part 2: Making Changes and Syncing (4 minutes)

1. Add and commit the analysis script:
    ```bash
    cp ../lectures/01/demo/files/vitals_analysis.py code/
    ```
    - In VS Code, the new file will appear in Source Control
    - Stage the file by clicking the + icon
    - Enter commit message: "Add patient vitals analysis script"
    - Click the checkmark to commit
    - Click "Sync Changes" to push to GitHub

2. GitHub repository features on the web:
    - View files and their history in the Code tab
    - See commit history in the Commits tab
    - Track tasks in the Issues tab
    - Configure repository in Settings
    - Review and merge changes in Pull requests
    - Edit files directly:
    * Click on any file to view it
    * Click the pencil icon to edit
    * Make changes in the web editor
    * Add a commit message
    * Choose to commit directly or create a pull request
    * Click "Commit changes"
    - Create new files:
    * Click "Add file" button
    * Choose "Create new file"
    * Enter filename (e.g., "data/notes.md")
    * Add file content
    * Add commit message
    * Choose branch
    * Click "Commit new file"

3. Pull changes from GitHub:
    - In VS Code, click "Source Control"
    - Click the "..." menu
    - Select "Pull" to get latest changes
    - Or use the sync button to pull and push

### Common Git Issues and Solutions

1. "Repository not found":
    - Check repository URL
    - Verify GitHub credentials
    - Try `git remote -v` to check remotes

2. "Changes not staged":
    - Use `git status` to see changes
    - Stage files in VS Code
    - Commit before pushing

3. Merge conflicts:
    - Pull latest changes first
    - Resolve conflicts in VS Code
    - Commit resolved changes

4. "Authentication failed":
    - Check GitHub credentials
    - Use personal access token if needed
    - Verify SSH key setup if using SSH

### Quick Exercise (2 minutes)

> Choose one of these exercises:
> 1. Create a new repository for your health data project and push your first commit
> 2. Clone an existing health data repository, make a small change, and push it back

## Demo 3: Python for Health Data Analysis (10 minutes)

### Introduction (1 minute)

> Python has become the language of choice for health data science due to its readability and powerful libraries. Let's explore how to use Python for health data analysis.

### Part 1: Running the Analysis Script (4 minutes)

1. Copy and run the script:
    ```bash
    cp ../lectures/01/demo/files/vitals_analysis.py code/
    python3 code/vitals_analysis.py
    ```

2. Explain the output:
    - Patient analysis results showing age, blood pressure, and BMI
    - Blood pressure categories (Normal, Elevated, Stage 1/2 Hypertension)
    - BMI calculations with example values

### Part 2: Understanding the Code (4 minutes)

The script uses these key Python concepts for health data analysis:

1. **Functions and Docstrings**:
    ```python
    def analyze_blood_pressure(systolic, diastolic):
    """Analyze blood pressure readings and return category."""
    if systolic < 120 and diastolic < 80:
        return "Normal"
    # ... more conditions ...
    ```
    - Functions encapsulate reusable code
    - Docstrings explain what functions do
    - Clear naming makes code self-documenting

2. **Data Types and Structures**:
    ```python
    # Numeric data
    age = 65                    # Integer
    temperature = 98.6          # Float
    
    # Collections
    heart_rate = [72, 75, 70]   # List of measurements
    patient = {                  # Dictionary for structured data
    "id": "A12345",
    "age": 65,
    "conditions": ["hypertension", "arthritis"]
    }
    ```
    - Different types for different data
    - Lists for sequences of values
    - Dictionaries for labeled data

3. **Control Flow and Analysis**:
    ```python
    if temperature > 100.4:
    print("Fever detected")
    else:
    print("Temperature normal")
    ```
    - Conditional logic for medical decisions
    - Loops for processing multiple values
    - Functions for complex calculations

4. **String Formatting**:
    ```python
    print(f"Patient age: {patient['age']}")
    print(f"Average heart rate: {average_heart_rate:.1f}")
    ```
    - f-strings for readable output
    - Formatting for precision control
    - Combining text and variables

### Error Handling Examples

If you encounter errors:
1. Syntax errors:
    - Check for missing colons or parentheses
    - Verify indentation
    - Look for typos in variable names

2. Runtime errors:
    - Check data types match function expectations
    - Verify file paths are correct
    - Ensure all required modules are imported

### Quick Exercise (2 minutes)

> Modify the vitals analysis script to add a new function that analyzes medication counts and returns a risk category based on the number of medications. Then run it to see the results."

## Final Thoughts

These demos are designed to build on each other, creating a cohesive learning experience:

1. **Command Line Demo**: Establishes basic skills for file management and simple data operations
2. **Git & GitHub Demo**: Builds on file management to introduce version control and collaboration
3. **Python Demo**: Leverages the previous skills to perform actual data analysis

Each demo includes health-specific examples to make the content relevant and engaging for health data science students. 