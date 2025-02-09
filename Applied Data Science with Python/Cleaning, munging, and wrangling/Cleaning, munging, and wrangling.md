# `data.clean()`

_if only it were that easy…_

- X-ray vision
- Packages - the best code (you didn’t write)
- Care and feeding of pandas
- Cleaning, munging, and wrangling
- Putting labels on things
- Getting dirty with MNIST

# 🦴 X-ray vision

![[data_quality.png]]

## Pre-trained X-ray model

- [https://github.com/mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision) (corresponding author at Stanford)
- CheXpert dataset ([paperswithcode](https://paperswithcode.com/dataset/chexpert), [kaggle-like competition](https://stanfordmlgroup.github.io/competitions/chexpert/))

![[xray1.png]]

![[xray1_attention.png]]

![[xray2.png]]

![[xray2_attention.png]]

  

> _Frontal and lateral radiographs of the chest in a patient with bilateral pleural effusions; the model localizes the effusions on both the frontal (top) and lateral (bottom) views, with predicted probabilities p=0.936 and p=0.939 on the frontal and lateral views respectively_

Let’s walk through an advanced task of object detection in an image by asking the pre-trained model to identify structures within an x-ray it hasn’t seen before. We’ll do this example using [Google Colab](https://www.notion.socolab.research.google.com).

1. Load GitHub repo (copy/paste [the URL](https://github.com/mlmed/torchxrayvision)) and select `segmentation.ipynb`
2. Add a line to install the missing library `%pip install torchxrayvision`
3. Run lines until we get to the image
4. Find and save a chest x-ray
    1. I used [this one](https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg)
    2. Upload to Colab, maybe in the /home directory
5. Run the model on that image

# 📦 Packages - the best code (you didn’t write)

## Quick reminder about _virtual environments_

Python works with environments to allow for multiple versions and settings for different projects. You can install libraries within a virtual environment without worrying about having them around and introducing incompatibilities within your default environment.

Depending on whether you’re using `conda` or native `python`, they work a little differently:

- The default environment
    - **conda** - The default environment in conda is called the `base` environment
    - **python** - The default environment otherwise is called the `system` environment
- Create an environment
    - **conda** - `conda create -n <NAME> <LIST OF PACKAGES TO INSTALL>`
        - Conda environments are _global_, they are created wherever conda is installed and can be used with any project regardless of location on the disk
    - **native** - `python3 -m venv <NAME> <LOCATION>`
        - Python environments are _local_, meaning they are created with a specific location and are most often tied to a project in the same location
        - NOTE: you may be able to use `python` instead of `python3`, but it is safer to specify python3 in case there is an old (2.x) version of python on your system
- Activate and deactivate environments
    - **conda**
        - Activate: `conda activate <NAME>`
        - Deactivate: `conda deactivate`
    - **native**
        - Activate:
            - Mac: `source <LOCATION>/bin/activate`
            - Win: `.\venv\Scripts\activate`
        - Deactivate: `deactivate`

## Installation

Installing packages is fairly easy and you can do it from inside a notebook:

- Installation
    - **conda** - `conda install <PACKAGE>`
    - **native** - `pip install <PACKAGE>`
- Update installed
    - **conda** - `conda update <PACKAGE>` or `conda update --all` to update all packages in the current environment
    - **native** - `pip install -U <PACKAGE>`

To install from a notebook, prefix the command with a `%`:

`%pip install <PACKAGE>`

`%conda install <PACKAGE>`

![[conda_install.png]]

### Packages used today

Take a moment to remember how virtual environments work, then make sure they’re installed

- `pandas` - importing and handling data
- `numpy` - large, multidimensional arrays
- `emnist` - dataset of handwritten digits to use in a classifier
- `matplotlib` - plot absolutely anything
- `seaborn` - slicker visuals with easy presets
- Later
    - `ydata-profiling` or `pandas-profiling` - automated exploratory data analysis
    - `sciki-learn`, `pytorch`, `tensorflow`, `keras` - ML frameworks

## Using packages

To use a package, we have to `import` it within the python code we’re running first.

You can import an entire library and all of the functions it contains :

```Python
import numpy as np
import pandas as pd

# Use NumPy and Pandas starting here
df = pd.DataFrame()
```

Or import only the functions you plan on using with `from` statements

```Python
from pandas import Series, DataFrame

# Does the same as the code block above
df = DataFrame()
```

## Quick reminder on using VS Code and Jupyter

### Kernels

Jupyter uses python to run commands and refers to available environments as _kernels_. When opening a notebook, make sure it is using the correct kernel. In VS Code, you can find the kernel at the top right of the notebook:

You can select from the available kernels by clicking on the current one, or using the “Notebook: Select Kernel” from the command palette:

![[Untitled_5.png]]

![[kernel_select.png]]

Whenever you install new packages, you may get a warning:

![[kernel_restart.png]]

To restart the kernel, use the Restart button:

![[kernel_restart_button.png]]

Or use the %magic command

```Python
%reset -f
```

### Common commands

> [!important]  
> These commands are for MacOS, on Windows use ctrl instead of cmd  

Common commands that you’ll need:

- **Command Palette** - `cmd-shift-P` or `ctrl-shift-P`
- “**Create: New Jupyter Notebook**” - easy way to make a new, blank notebook from the command palette

- **Run Cell** - `shift-enter` or click the button labeled “Run” or ▶️ to run just that one cell

![[jupyter_run_cell.png]]

- **Run All** - button to run all cells, stopping at the first error. Useful when using
    - **NOTE: “**Run All” doesn’t clear anything you’ve done from memory, which _will_ get you into trouble. **To run everything from a clean slate: Restart > Run All**
- **Clear All Outputs** - Good idea to do this before **committing** your changes in git

- **Comment** - Comments start with `#` and are not run as code. `cmd-/` will turn the current line into a comment

![[inline_comment.png]]

- **Add Cells** - Use the `+` buttons to add Code and Markdown cells. Don’t forget to document what you’ve done with Markdown!

![[add_cells.png]]

- **git** - the Source Control button on the toolbar can do everything you need with git: **init**ialize or **clone** a repo, **stage** changes (add files with the `+` button), **commit** your changes (be sure to add a message), and **sync** with the cloud.

![[source_control.png]]

- **terminal** - `` ctrl-` `` will open a terminal within VS Code, you may need to activate the correct environment once open

# 🐼 Care and feeding of pandas

Pandas is the cutest named best friend you’ll have working with data. At times, it will also be your nemesis. Often it will be used with friends: NumPy, SciPy, scikit-learn, PyTorch, and TensorFlow.

There is a standard naming convention when importing pandas (and many other packages):

```Python
import pandas as pd
import numpy as np
```

## Need more?

- _Python for Data Analysis_, McKinney (ch5-9)- author’s [website](https://wesmckinney.com/book/)
- _Python Data Science Handbook,_ VanderPlas (ch3) - author’s [website](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Berkeley’s DS100’s](https://ds100.org/) first few lectures cover this in more detail  [https://ds100.org/](https://ds100.org/)

## Dataframes

Two objects you’ll use in Pandas are dataframes and series.

A **dataframe** is a two-dimensional matrix that can hold many types of values, even other dataframes, as its values. You can create one by manually defining each column and entering values.

A **series** is a one-dimensional objects, like a single row in a dataframe.

```Python
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
 "year": [2000, 2001, 2002, 2001, 2002, 2003],
 "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
```

This way, we can see what `data` (list of named lists) and `frame` (dataframe) look like. The dataframe looks a lot more approachable.

![[jupyter_data.png]]

![[jupyter_frame.png]]

## Loading data

We probably want more data than we can enter by hand. Most of the time that will mean tab-separated (.tsv), comma-separated (.csv), and Excel (.xlsx) documents.

Luckily, Pandas can read all of those into dataframes (sometimes without issues!)

- **read_csv** - to open tab- or comma-delimited files
- **read_clipboad** - copy without the paste! make a dataframe from the current contents of the clipboard
- **read_excel** - does what it says
- **read_html** - read all tables in an HTML document
- **read_json** - create dataframes from JavaScript Object Notation strings; this will prove incredibly valuable later on
- **read_pickle** - pickle (or .pkl) is a python-specific data format
- **read_parquet** - parquet is an incredibly versatile format, often used for BIG datasets

```Python
# Create a dataframe 'df'
df = pd.
df = pd.read_csv("<PATH TO FILE.CSV>")

# Now "df" is a dataframe with the contents of your CSV
```

Find details and examples [in the book](https://wesmckinney.com/book/accessing-data.html).

## Slicing and selecting

Two things about Pandas that don’t feel very natural when you first try them: filtering and merging. We won’t cover merging in this, but they are both fundamental skills that only get better with practice

### Get columns

Retrieve columns from a dataframe by listing the desired columns within double-brackets, returning a new dataframe with only those columns:

```Python
# Get state and year as a dataframe 
df[['state', 'year']]

# Get the state column as a dataframe
df[['state']]

# Get the stat column as a series
df['state']
```

![[get_cols.png]]

### Drop columns

Drop columns and rows by specifying which to get rid of. Again, this will create a new dataframe with the columns omitted. The original dataframe will remain untouched.

```Python
df.drop(columns=['state'])
```

![[drop_cols.png]]

  

### Get rows

```Python
# Get the first row of the dataframe
df[0]
```

### Filter rows by value

Include only the rows you want by specifying conditions within brackets:

```Python
frame[(frame['state'] == 'Ohio') & (frame['year'] == 2000)]
```

![[filter_cols.png]]

This can be useful for separating your data, perhaps focusing on one partition while leaving the other untouched. Note we didn’t need the parentheses when using only a single condition.

```Python
df_ohio = frame[frame['state'] == 'Ohio']
df_else = frame[frame['state'] != 'Ohio']
```

### String contents

Pandas has built-in tools for searching string values. These will come in handy when finding valid/invalid data in variables. You can also use regular expressions, but those are outside the scope for today.

- `str[0]` - check the first character of a string, `df[df['var1'].str[0] == 'A']`
- `str.len()` - length of the string, `df[df['var1'].str.len()>3]`
- `str.contains()` - search within, `df[df['var1'].str.contains('<PATTERN>')]`

### Quick note on special characters!

Searches can include special characters like quotes `'`, pipes `|`, and slashes `\` , but they must be _escaped_ by preceding with a `\`. For example, looking for “quoted text” (including the quotes) would be `contains('\"quoted text\"')`.

You can also use the `re.escape()` function to do this automatically:

```Python
import re
escaped_string = re.escape(string_with_special_characters)
```

Without escaping special characters, you might get different results than you intended. Searching for “A|B” with `contains('A|B')` matches any string with either “A”, “B”, or both. With the backslash before “|”, `contains('A\|B')` matches “A|B”.

### Negation

You can negate conditions by using `!=` instead of `==` , or by using `~` to negate a whole expression:

```Python
# Get the Ohio rows
df_ohio = frame[frame['state'] == 'Ohio']

# Get the NOT Ohio rows using a tilde, "~"
df_else1 = frame[~(frame['state'] == 'Ohio')]

# Get the NOT Ohio rows using not-equals, "!="
df_else2 = frame[frame['state'] != 'Ohio']
```

### Query

> [!important]  
> This is a mini-preview of next lecture’s topic: SQL!  

You can combine multiple conditions using the `query()` method. It will take the contents of a valid SQL `WHERE` clause and apply it to the dataframe. That may sound complicated, but it’s really handy once you get the basics of SQL.

- `.query(<EXPRESSION>)`

![[query_df.png]]

### Transpose

You can pivot a dataframe in one command with `transpose()`

```Python
frame.transpose()
```

![[df_transpose.png]]

# 🧼 Cleaning, munging, and wrangling

![[File_Feb_11_2023_1_25_05_PM.jpeg]]

The three terms are often used inconsistently interchangeably:

- **Cleaning** - handling duplicate, missing, invalid, or irrelevant data
- **Wrangling** - Transform, recode, and enrich
- **Munging** - Prepare the output for downstream systems

Or did I get _wrangling_ and _munging_ switched?

### Following along at home

We will walk through the Pandas and data-cleaning content from Wes McKinney’s _Python for Data Analysis_ (chapters 5-9)

### The data pre-processing process

In any case, the process goes something like this:

1. **Explore** the data set to understand what it contains
2. **Clean** out duplicates, outliers, and invalid data
3. **Recode** and transform individual variables
4. **Enhance** by combining variables and merging related data (_we will skip this_)
5. **Validate**, and return to earlier steps as needed
6. **Export** and document

## Exploration

Exploratory data analysis is the best way to get to know each other. EDA helps to reveal what the data contains and, with some prodding, what is in there that _isn’t quite right…_

First, get a quick view of some raw data and how much of it you’ve got:

- `.head()` and `.tail()` - show the first/last 5 (or more) rows
- `.shape()` - count the rows and columns
- `.info()` - type of columns
- `value_counts()` - how often values of a column occur, useful for categorical variables
- `.count()` - count of non NaN/null values

### `df.describe()`

The quickest way to get a sense of the contents of each variable.

You’ll probably need to do this at least twice! Once for the numerical variables and again for the categorical ones. Repeat as needed after cleaning attempts.

Running `.describe()` on a whole dataframe will return information **only** about its numerical columns:

```Python
frame.describe()
```

![[df_describe.png]]

To get a summary of categorical columns we have to specify those rows:

```Python
frame[['state']].describe()
```

![[df_describe_cat.png]]

### Bivariate analysis with `df.corr()` and `seaborn.pairplot()`

You can get quick visual and statistical analysis of the relationship between pairs of variables with `seaborn` (visual) and `df.corr()` (stats).

`df.corr()` calculates correlation coefficients between each pair of numerical variables

```Python
# To find the correlation among
# the columns using pearson method
df.corr(method ='pearson')
```

![[df_corr.png]]

- Python | Pandas dataframe.corr() ([geeksforgeeks](https://www.geeksforgeeks.org/python-pandas-dataframe-corr/))

  

`seaborn.pairplot()` plots all variables (numerical and categorical) against each other

```Python
# Seaborn visualization library
import seaborn as sns

# Create the default pairplot
sns.pairplot(dataframe)
```

![[sns_pairplot.png]]

- Visualizing Data with Pairs Plots in Python ([towardsdatascience](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166))

### ~~`pandas-profiling`~~ `ydata-profiling`

Cheating! Not really.

There are some good tools to help automate the process of Exploratory Data Analysis (EDA). One of them is `ydata-profiling` (formerly `pandas-profiling`, which is still the version in conda), which will generate a summary of your dataset to get you started more quickly.

- [https://github.com/ydataai/ydata-profiling](https://github.com/ydataai/ydata-profiling)
- Pandas Profiling: One-Line Magical Code for EDA ([[pand]])

```Python
# Install pandas_profiling
%conda install pandas_profiling

# Import
from pandas_profiling import ProfileReport

# Generate report
report = ProfileReport(frame)

# Save as html
report.to_file("profile_report.html")

# View in notebook
report.to_notebook_iframe()
```

![[pandas_profiling.png]]

  

## Cleaning duplicates, outliers, and invalid data

### Try to stay in shape

The easiest way to keep track of your data is to maintain the same number of rows during data cleaning.

As you go along and clean a dataset, please keep track of what data you plan to include and exclude **without deleting it**. You can create a new column to keep track of rows you plan to exclude later.

Always copy the last known good version of the data when you make changes. That way, you can revert to the “good version” if the changes don’t go as planned. In programming, things will often not go as planned.

### Finding missing values

Missing rows can take many forms. Python has a built-in value of `None` and Pandas adds `NaN`, but your data might have its own ideas.

Common forms of missing data are:

- `None` - Python built-in
- `NaN` - Pandas built-in
- “”, “Missing”, “null” - various strings (or an empty string) often denote missing
- 999999999 - nine 9’s is a common, older practice for encoding missing data

Once you know which values mean “missing”, replace those with ones that python will understand. Let’s say that a column uses the string “banana” to denote a missing value. We can `replace()` those bananas as `None`.

```Python
# Make a copy of the good dataframe to work with
work_df = good_df.copy()

# Replace those bananas with something a python can understand
work_df["column"] = work_df["columm"].replace("banana", np.nan)

# We can also replace the column in-place
work_df["columm"].replace("banana", np.nan, inplace=True)
```

  

### Handling the missing

What to do about missing values? That depends. How important is it that the column be accurate for your analysis? If they must be accurate for the analysis to succeed, then your best bet may be to filter out the missing. If some inaccuracy could be acceptable, then a filling/replacement strategy could be best.

- **Filtering** - `df.dropna(<how="method">)` dropna will return
- **Filling** - `df.fillna()` replaces missing values. There are many ways that it can do this interpolation, ranging from the very simple (replace with 0) to highly complicated

### Duplicates

Pandas makes this one really easy with the `duplicated()` method, which returns a list of `True` and `False` values; one for each row indicating if it is a duplicate or not. We can add even add that list as a new column in our dataframe!

```Python
# Check for duplicated rows across all columns
df.duplicated()

# Check for duplication within a subset of columns, keep the first
df.duplicated(subset=['col1', 'col2'], keep='first')

# Save the duplication list for 'image' as a new column
df['image_isdup'] = df.duplicated(subset=['image'])
```

### Outliers and out-of-bounds

Numerical data may not always make sense. Extreme values may indicate rows that are not representative and should be excluded (e.g., 8 feet tall). Some variables may have a finite range, so values outside that range would be nonsensical (e.g, -2 feet tall, probability less than 0 or greater than 1)

### Labels and strings

Details in ch7 of the book and far more at:

- String Manipulation in Python ([pythonforbeginners](https://www.pythonforbeginners.com/basics/string-manipulation-in-python))
- Python String Manipulation Handbook ([freecodecamp](https://www.freecodecamp.org/news/python-string-manipulation-handbook/))

  

There’s a **LOT** here, we will only touch on it:

- `replace('this', 'that')` - replace ‘this’ with ‘that’ within a string (or regex, list, dict, Series, int, float)
- `split()`, `strip()`, and `join()`
    - `split()` separates parts of a string using a delimiter like a comma “,”
    - `strip()` removes leading and trailing whitespace
    - `join()` combines the pieces back together
- `lower()` and `.upper()` - convert to upper- or lowercase
- `find()` and `rfind()` - return the position of the first character of the first occurrence (last occurrence for `rfind`) of a substring within a string
- `startswith()` and `endswith()` - return True if string starts/ends with substring
- `count()` - returns the number of non-overlapping occurrences of a substring within a string

## Transform and recode

### One-hot dummy

The `.get_dummies()` method returns columns with one-hot dummy variables for a categorical column

```Python
# Suppose the key column had values a, b, c
# dummies now has columns key_a, key_b, and key_c
# key_a will be 1 for rows when key == A
dummies = pd.get_dummies(df["key"], prefix="key")

# Then we can add those columns 
df_with_dummies = pd.concat([df, dummies], axis=1)
```

### Binning / discretization

Pandas provides a few ways to bin data from a column:

- `cut()` - allows you to set cut points and labels for them

```Python
# Assign grades with cutpoints of 0, 50, 80, and 100
bins = [0, 50, 80, 100]
labels = ['C', 'B', 'A'] # C = 0-50, B=50-80, A=80-100
df['grade'] = pd.cut(x = df['score'], bins = bins, labels = labels, include_lowest = True)
```

- `qcut()` - divide the data into equal-sized buckets based on rank

```Python
# Assign grades A, B, and C to an equal number of students
df['grade'] = pd.qcut(df['score'], q = 3, labels = ['C', 'B', 'A'])
```

- `value_counts()` - we used this above for EDA, but it also tell us the values for defining bins

```Python
# Define three bins of equal width
df['score'].value_counts(bins = 3, sort = False)

# Returns bins of (0, 33.33], (33.33, 66.66], (66.66, 100]
```

### Category reclassification

Categorical variables may not match the groupings needed for analysis. For examples, there may be many small categories that should be grouped together as “Other” or the categories may be at too fine a granularity.

Grouping small categories together as “Other”

```Python
# Use value_counts() to find labels that occur less than a threshold
# We'll use 9999 as our threshold here

# Get the counts of all labels
counts = raw_train['label'].value_counts()

# Find the small categories
index = counts.lt(9999).index

# Relabel using loc to match against the small categories
# The follow two lines are equivalent
work['new'][work['new'].isin(small_cats)] = 'Other'
work['new'] = np.where(work['new'].isin(small_cats), 'Other', work['new'])



# See "Mapping" for how to map specific categories into "Other"
```

### Mapping

Pandas allows you to `map`( between string values that exist in the data and the values that _should_ be in the data using a dictionary (or `dict`). This could be very useful if your professor gives you a dataset in a few sections where the labels are all numbers, but they actually correspond to the numbers 0-9 and upper-/lower-case letters.

```Python
# Define our dictionary
animal_sounds = {
	"cat": "meow",
	"dog": "bark"
}

df["sound"] = data["animal"].map(animal_sounds)
```

## Are we done yet? Validate!

Time to see if we’re done. Repeat the exploratory data analysis (EDA) that you used to find the dirty data in the first place.

Mark any rows that should be excluded and create clean copies of your **train** and **test** data.

# 🦾 Exercise

1. (Highly recommended) Work in small groups using **branches** on the same repo in GitHub
2. (Recommended) Practice exercises in the practice folder (from [https://ds100.org](https://ds100.org/))
3. Coding submission: 👇

## 🪨 Getting dirty with MNIST

MNIST is the “Hello, World!” of image classification. The original dataset, “**NIST Special Database 19**”, sounds ominous but is just a small set of low-resolution (28x28 pixels), hand-written digits. We will use a larger dataset, [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), with alphanumeric characters. It’s pretty big, so you might want to install grab it in whatever notebook you plan on using for the exercises.

```Python
# Import EMNIST, install using `pip` (conda doesn't have the package)
import emnist

raw_train[['image', 'label']] = emnist.extract_training_samples('byclass')
raw_test[['image', 'label']]  = emnist.extract_test_samples('byclass')
```

- Lots of examples using the data [under the Code tab on kaggle](https://www.kaggle.com/datasets/crawford/emnist/code)

### First, let’s mess up the data

I’ve provided a notebook that will pull the EMNIST and muck it up a little at random, so we can get it clean again.

[The notebook to mess things up is on GitHub](https://github.com/christopherseaman/datasci-seminar)

  

Things I definitely did **NOT** do:

- Add noise - Dirty-MNIST ([paperswithcode](https://paperswithcode.com/dataset/dirty-mnist))
- Mix in other data (pictures of animals instead of symbols)

### Now we clean it

This is all for fun (not credit), so feel free to group up. Divide and conquer! Share one version of the dirty dataset, then work with a shared repo and branches.

Things to do:

- Get familiar with the data
- Merge the test and train data (keep track of which is which!)
- Fix the labels so they are more meaningful
    - The labels in this dataset are numbered 1-62 and correspond to the digits 0-9, lowercase a-z, and uppercase A-Z
    - Try to add a column with human-readable labels
    - Why might the data have been labeled the way it was?
- Cleaning
    - Duplicate rows
    - Missing values
    - Outliers and out-of-bounds issues
    - Label issues
    - Image issues
        - Zeroed
        - ~~Wrong dimensions~~

# Further reading

## Data munging and pandas

- _Python for Data Analysis_, McKinney (ch5-9)- author’s [website](https://wesmckinney.com/book/)
- _Python Data Science Handbook,_ VanderPlas (ch3) - author’s [website](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Berkeley’s DS100’s](https://ds100.org/) first few lectures cover this in more detail  [https://ds100.org/](https://ds100.org/)
- Critical Field Guide for Working with ML Datasets ([knowingmachines](https://knowingmachines.org/critical-field-guide))
- How to Prepare Your Dataset for Machine Learning and Analysis ([jetbrains](https://blog.jetbrains.com/datalore/2022/11/08/how-to-prepare-your-dataset-for-machine-learning-and-analysis/))
- Pythonic Data Cleaning With pandas and NumPy ([realpython](https://realpython.com/python-data-cleaning-numpy-pandas/))

> [!info] Data Cleaning Plan  
> Data cleaning or data wrangling is the process of organizing and transforming raw data into a dataset that can be easily accessed and analyzed.  
> [https://cghlewis.github.io/mpsi-data-training/training_4.html](https://cghlewis.github.io/mpsi-data-training/training_4.html)  

## Classification approaches prepping for next topic

Write-ups that are definitely applicable:

- 🌟 Supervised & Unsupervised Techniques on MNIST ([medium](https://medium.com/@muhammetbolat/supervised-unsupervised-techniques-on-mnist-dataset-3f2ffd4c41c5))
- Unsupervised Learning for MNIST with EDA ([kaggle](https://www.kaggle.com/code/manabendrarout/unsupervised-learning-for-mnist-with-eda/notebook))
- Exploring EMNIST - another MNIST-like dataset ([simonwenkel](https://www.simonwenkel.com/notes/ai/datasets/vision/EMNIST.html))
- PyTorch
    - Handwritten Digit Recognition Using PyTorch ([medium](https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627))
- Keras / TensorFlow
    - 🌟 Training a neural network on MNIST with Keras TensorFlow ([google](https://www.tensorflow.org/datasets/keras_example))

  

### Also related:

- A dataset of tiny images ([Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch), ch7.1)
- MNIST using RandomForest + Scikit-Learn ([github](https://richcorrado.github.io/MNIST_Digits-overview.html))
- R and Keras: Build a Handwritten Digit Classifier ([appsilon](https://appsilon.com/r-keras-mnist/))