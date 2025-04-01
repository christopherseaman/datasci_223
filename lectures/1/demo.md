# Demo 1: Command Line Basics for Health Data Science

<!--
SPEAKING NOTES:
- Demo objectives: Basic navigation, file operations, and text manipulation
- Expected outcome: Students should be able to navigate directories, create/edit files, and use basic text tools
- Troubleshooting: Watch for common mistakes like typos in commands or paths
- Validation: Ask students to try a simple command on their own systems
-->

Let's explore the command line together:

1. Opening the terminal
   - Mac: Terminal app
   - Windows: WSL Ubuntu terminal
   - Linux: Terminal app

2. Basic navigation
   - `pwd` - Where am I?
   - `ls -la` - What's here? (including hidden files)
   - `cd` - Moving around
   - Special directories: `~` (home), `.` (current), `..` (parent)

3. Creating a health data project structure
   - `mkdir health_project`
   - `cd health_project`
   - `mkdir data code results documentation`
   - `touch README.md`
   - `ls -la`

4. Viewing and editing files
   - `echo "# Health Data Analysis Project" > README.md`
   - `cat README.md`
   - `nano README.md` (add a description)
   - `cat README.md`

5. Working with health data examples
   - Create a sample data file: `touch patient_data.csv`
   - Add some content: `echo "id,age,condition" > patient_data.csv`
   - Add more rows: `echo "1,45,hypertension" >> patient_data.csv`
   - View the file: `cat patient_data.csv`

6. Searching and filtering health data
   - Search for a term: `grep "hypertension" patient_data.csv`
   - Count lines: `wc -l patient_data.csv`
   - Combine commands: `cat patient_data.csv | grep "hypertension" | wc -l`

> **Quick Exercise**: Try creating a file called `notes.txt` with a few lines about what you hope to learn, then use `grep` to find specific words in it.

# Demo 2: Git and GitHub with Codespaces for Health Research

<!--
SPEAKING NOTES:
- Demo objectives: Setting up Git, creating repositories, and using GitHub Codespaces
- Expected outcome: Students should understand basic Git workflow and be able to use GitHub Codespaces
- Troubleshooting: Watch for common issues with Git configuration and authentication
- Validation: Ask students to create their own repository and make a simple commit
-->

Let's explore Git and GitHub Codespaces:

1. GitHub account setup
   - Creating an account
   - Setting up GitHub Education benefits
   - Configuring your profile
   - Setting up a secure email address (privacy considerations)

2. Creating a health research repository
   - New repository on GitHub
   - Adding a README.md with project description
   - Adding a .gitignore for Python
   - Adding a LICENSE file (important for research code)

3. Using GitHub Codespaces
   - Launching a Codespace from your repository
   - Exploring the VS Code interface in the browser
   - Installing extensions relevant to health data science
   - Understanding persistence and saving work

4. Making changes with Git
   - Creating a new file (e.g., `data_cleaning.py`)
   - Staging changes with git add
   - Committing changes with a descriptive message
   - Pushing changes to GitHub
   - Viewing commit history

5. Branching and merging for collaborative research
   - Creating a feature branch (e.g., `add-visualization`)
   - Making changes on the branch
   - Creating a pull request with detailed description
   - Reviewing changes (code review process)
   - Merging the changes
   - Discussing when to use branches in research projects

> **Quick Exercise**: Create a new repository with a README.md file that describes a health data science project you're interested in.

# Demo 3: Python Basics for Health Data Science

<!--
SPEAKING NOTES:
- Demo objectives: Basic Python syntax, variables, and simple data manipulation
- Expected outcome: Students should understand basic Python syntax and how to run simple scripts
- Troubleshooting: Watch for common syntax errors like missing colons or indentation issues
- Validation: Ask students to modify a variable and see how it changes the output
-->

Let's explore Python in different environments:

1. Running Python
   - In the terminal: `python3`
   - In VS Code/Codespaces
   - In Jupyter notebooks (preferred for exploratory health data analysis)

2. Basic Python syntax with health data examples
   - Variables and data types:
     ```python
     patient_name = "Jane Doe"  # String (text) - always anonymized for teaching
     patient_age = 65           # Integer (whole number)
     blood_glucose = 140.5      # Float (decimal number)
     has_diabetes = True        # Boolean (True/False)
     ```
   - Simple operations:
     ```python
     # Calculate BMI
     weight = 70  # kg
     height = 1.75  # meters
     bmi = weight / (height ** 2)
     print(f"BMI: {bmi:.1f}")
     ```
   - Control flow (if/else, loops):
     ```python
     # Categorize blood pressure
     systolic = 135
     if systolic < 120:
         print("Normal")
     elif systolic < 130:
         print("Elevated")
     else:
         print("High")
     ```
   - Functions:
     ```python
     def calculate_bmi(weight, height):
         return weight / (height ** 2)
     ```

3. Creating a health statistics calculator
   - Create file: `nano health_stats.py`
   - Add code:
     ```python
     # Simple health statistics calculator
     height = float(input("Enter height in cm: "))
     weight = float(input("Enter weight in kg: "))
     
     bmi = weight / ((height/100) ** 2)
     print(f"BMI: {bmi:.1f}")
     
     if bmi < 18.5:
         print("Category: Underweight")
     elif bmi < 25:
         print("Category: Normal weight")
     elif bmi < 30:
         print("Category: Overweight")
     else:
         print("Category: Obesity")
     ```
   - Run script: `python3 health_stats.py`

4. Virtual environments for reproducible health research
   - Creating a virtual environment: `python3 -m venv health_env`
   - Activating: `source health_env/bin/activate` (Mac/Linux) or `health_env\Scripts\activate` (Windows)
   - Installing packages: `pip install pandas numpy matplotlib`
   - Creating requirements.txt: `pip freeze > requirements.txt`
   - Deactivating: `deactivate`

5. Jupyter notebook for exploratory health data analysis
   - Launch Jupyter: `jupyter notebook`
   - Create new notebook
   - Show code and markdown cells
   - Demonstrate running cells and viewing output
   - Show how to document analysis steps

> **Quick Exercise**: Try modifying the BMI calculator to add one more category or to calculate another health metric like body surface area.

# Demo 4: Hands-on in the cloud with Colab for Health Data Science

<!--
SPEAKING NOTES:
- Demo objectives: Using Google Colab for Python notebooks and connecting to GitHub
- Expected outcome: Students should be able to open, edit, and save notebooks in Colab
- Troubleshooting: Watch for GitHub authentication issues and Colab connection problems
- Validation: Ask students to make a simple edit to a notebook and commit it
- IMPORTANT: Remind students never to use PHI data in Colab
-->

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

8. Add a health data science example to the notebook:
   ```python
   # Health data example
   patient_data = {
       'patient_id': [1, 2, 3, 4, 5],
       'age': [45, 62, 35, 28, 71],
       'condition': ['hypertension', 'diabetes', 'asthma', 'migraine', 'arthritis'],
       'medication_count': [2, 3, 1, 1, 4]
   }
   
   # Calculate average age
   avg_age = sum(patient_data['age']) / len(patient_data['age'])
   print(f"Average patient age: {avg_age:.1f}")
   
   # Find patients with multiple medications
   multiple_meds = [i for i, count in enumerate(patient_data['medication_count']) if count > 1]
   print(f"Patients with multiple medications: {[patient_data['patient_id'][i] for i in multiple_meds]}")
   ```

9. Run cells by clicking the "play" icon at top left of it, or by hitting shift+enter while the cell is selected
   - A scary warning will pop up the first time you run a cell. You can dismiss it by clicking "Run anyway".
   - It is possible that a malicious notebook could request access to your data, but this one should be safe.

10. Commit changes you've made to GitHub directly from Colab. The "Cannot save changes" warning is misleading. To commit changes:
    - Click File > Save a copy in GitHub (you may need to sign in to GitHub)
    - This should open a dialogue with the repo, branch, and file name of the notebook you're working in
    - Edit the commit message and click "OK"

> **Quick Exercise**: Add a visualization to the patient data example using matplotlib.
