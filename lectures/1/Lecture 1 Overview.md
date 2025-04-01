# Lecture 1 Overview

### Virtual Environments

pathway doesn’t need to be in project folder

```jsx
cd path/to/directory
python3.11 -m venv .venv_name
source .venv_name/bin/activate

#to get pip updated/installed
python3 -m pip install --upgrade pip
python3 -m pip --version

#install pkg_name (eg jupyter, numpy)
python3 -m pip install pkg_name

pip install ipykernel
# let notebook know about the venv
python3 -m ipykernel install --user --name=myenv1 
```

```jsx
cd path/to/directory
python -m venv .venv_name
.\.venv_name\Scripts\activate.bat

#to get pip updated/installed
py -m pip install --upgrade pip
py -m pip --version

#install pkg_name
py -m pip install pkg_name
```

<aside>
❗ It’s not recommended to upload your virtual environment to github, so be sure to remove this file when you are uploading to github
`git rm .venv_name`

</aside>

- want to export current packages → `pip freeze > requirements.txt`
- if you already have requirements file → `pip install -r requirements.txt`
- when done →`deactivate`

---

### Git Review

`git clone` is basically a combination of:

- `git init` (create the local repository)
- `git remote add` (add the URL to that repository)
- `git fetch` (fetch all branches from that URL to your local repository)
- `git checkout` (create all the files of the main branch in your working tree)

if you do `git init` in a directory that's not empty, `git clone` will abort with `"fatal: destination path '...' already exists and is not an empty directory.”`

**Forking a Repository**

Go to repo on github.com/user/repo-name.git

Click “Fork” and rename if you’d like → Click on **Create Fork** when done

Then copy link under **<>Code (https)**

![Screenshot 2024-01-11 at 1.01.00 PM.png](Lecture%201%20Overview%2044c1d71be20f4937a84dbe031c15cbfa/Screenshot_2024-01-11_at_1.01.00_PM.png)

```jsx
**# Set up your local repository**
cd path/to/directory #where you want your repo folder to be
git init
git clone https://github.com/your-username/forked-repo-name.git

**# Set upstream to the original repository**
git remote -v

#if nothing shows 
remote add origin https://github.com/your-username/forked-repo-name.git

#else:
git remote add upstream https://github.com/original-user/repo-you-copied.git
git remote -v

**# Update your fork with changes from upstream**
git checkout -b branch-name 
git pull upstream main #pulls main branch from upstream (original repo)

**# Make and commit changes on the branch**
git status **#check what changes**
git add . # **.** for all, or "file-name"
git status #I like to do this again to check that I added the right files

#if you added something wrong or made more changes before committing
git reset

#then run 
git commit -m "Your commit message"

**# Push the new branch to your fork**
git push -u origin branch-name  # sets up tracking for future pushes

**# Make and commit changes on the branch**
# ...

**# Push changes to your fork**
git push #for following pushes on that fork
git push origin branch-name (if different fork)

**# Update the main branch with changes from the feature branch**
git checkout main
git merge branch-name

**# Delete the feature branch**
git branch -d branch-name  # use -D if changes are not merged
```

```jsx
**# Chris updated his datasci 223 repo?**
git checkout branch-name #to have a branch to store your changes
git stash #store these changes on this branch
****
git checkout main #switch to YOUR main branch
git pull upstream main #update it with upstream's main branch

**# If you want to update your local branch with changes online**
git pull --ff-only #fast forward only, since git pull origin main means merging

**# If you had changes on feature branch (after pulling)**
git checkout branch-name
git merge main  # or **git rebase main** so now changes on dev branch are on top of it

**# then merge those changes on dev branch to main**
git checkout main
git merge branch-name
```

---

### Jupyter Notebooks

- For **activating kernel** in the **jupyter noteboo**k within VS code
    - A virtual environment is an isolated copy of a python installation that can be used to run Jupyter.
    - If you create a python 3 virtual env, you can install Jupyter into this environment using pip and then launch Jupyter.
    - The only python "kernels" it will then show will be the python virtual environment. A kernel in Jupyter is a different engine for running notebook cells in iPython. If you install packages in a python virtual env you get those packages only in the kernel that virtual environment can give you
    
    Mac: `cmd-shift-P` , Windows: `ctrl-shift-P`  → Select Notebook Kernel **or** 
    
    ![Screenshot 2024-01-10 at 11.03.02 PM.png](Lecture%201%20Overview%2044c1d71be20f4937a84dbe031c15cbfa/Screenshot_2024-01-10_at_11.03.02_PM.png)
    
    - Once you’re ready to commit changes → **Clear All Outputs**:
    
    ![Untitled](Lecture%201%20Overview%2044c1d71be20f4937a84dbe031c15cbfa/Untitled.png)
    

Extra notes/Q’s

- **How do I make a Markdown document?** Use cmd-N/ctrl-N as
normal to create a new file, start typing, then save with the correct extension. (Save as: “README.md, Format: All Formats)
- control + R for previous commands on terminal