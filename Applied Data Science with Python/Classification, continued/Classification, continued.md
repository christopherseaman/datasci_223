# `git dangit`

## Avoiding `git merge` conflicts with branches

1. Working in branches for each exercise
2. Save to branch and sync (discarding commits)
3. (_Optional_) Add files from branch back to `main` through a Pull Request

![[git_branches.png]]

## Example:

1. Save the work I have in `main` to `branch1`

```Shell
git checkout -b branch1  # Create and switch to branch1
git add .               # Add all changes
git commit -m "Saving changes before syncing with upstream"
```

1. Discard changes on `main` and rebase from upstream

```Shell
git checkout main  # Switch to the main branch
git reset --hard upstream/main  # Reset main to match upstream/main (assuming 'upstream' is your upstream remote)
```

1. Pull in specific files from `branch1` using `git checkout branch1 -- <file_path>`

```Shell
git checkout main  # Switch to the main branch if we hadn't already
git checkout branch1 -- exercises/4-classification/exercise.ipynb # Pull changes from branch1 for specific exercise file
git add exercises/4-classification/exercise.ipynb # Add to the staging area
git commit -m 'exercise 4 added to main branch' # Commit changes to main branch
```

1. Delete `branch1` once changes are saved to `main`

```Shell
git branch -d branch1  # Delete branch1
```

## `git reset` and `git revert`

> [!important]  
> git reset is DESTRUCTIVE! It is used to delete changes to your git commit history  

You can reset Git to any commit with:

- `git reset @~N` or equivalently `git reset HEAD~N`

Where N is the number of commits before HEAD, and `@~`

resets to the previous commit. So if we're here:

```Plain
A <- B <- C
          ^ HEAD
```

- `git reset @~` undoes one commit, setting HEAD to B
- `git reset @~2` undoes two commits, setting HEAD to A

### Resetting to a specific commit

- `git reset <HASH>` undoes commits back to the commit defined by <HASH>

You can get the short hash of each commit using `git reflog`

```Shell
❯ git reflog
0728ee2 (HEAD -> main) HEAD@{0}: checkout: moving from classification-+-fhir to main
b4d3760 (classification-+-fhir) HEAD@{1}: commit: sync
fde9224 HEAD@{2}: pull --tags origin classification-+-fhir: Fast-forward
f6e0c45 HEAD@{3}: checkout: moving from main to classification-+-fhir
0728ee2 (HEAD -> main) HEAD@{4}: checkout: moving from classification-+-fhir to main
f6e0c45 HEAD@{5}: checkout: moving from main to classification-+-fhir
0728ee2 (HEAD -> main) HEAD@{6}: commit: Lecture 4 update
```

In this case, `git reset 0728ee2` would take the HEAD back to “Lecture 4 update”

### Types of `git reset`

Resets may be soft, hard, or mixed. Generally, I find mixed (default) or hard most useful:

- `git reset @~ --soft` : Moves the commit back one, leaves all changed files in the working directory, they are staged for commit
- `git reset @~ --mixed` : Moves the commit back one, leaves all changed files in the working directory, they are NOT staged for commit
- `git reset @~ --hard` : Moves the commit back one, discards all changed files in the working directory since that commit

### Versus `git revert`

`git revert` is somewhat similar, but works differently. It adds a new commit that undoes the changes of the previous commit. All of your changes are still included in the repository history (which would be large in your case). In this case it would make a new commit B' that looks the same as B, but the .git folder would include all the history and files from C

```Plain
A <- B <- C <- B'
               ^ HEAD
```

# Systematic model selection

1. **Preparation and Setup**
    - **Parameter Definition:** Establish the number of splits or folds (K) for the validation process, decide on the repetitions for techniques like Repeated K-Fold or K-Split if needed, and outline the hyperparameter grid (C) for each candidate model. For models lacking hyperparameters, set C to an empty configuration.
2. **Data Partitioning**
    - **Validation Holdout (Optional):** Optionally reserve a subset of the data as a standalone validation set (V) to provide an unbiased final evaluation.
3. **Model Evaluation**
    - **K Splits/Folds Validation (Outer Loop):** Organize the data into K distinct splits or folds, using a consistent strategy to ensure each segment of data is used for validation once. This could involve stratified sampling to preserve class distributions in cases of imbalanced datasets.
        - **Hyperparameter Optimization (Inner Loop):** Within each split or fold, perform hyperparameter tuning for models with configurable parameters. This can include nested validation within the training portion of each split or fold to determine the optimal settings.
4. **Performance Aggregation**
    - **Score Compilation:** Collect and average the performance scores across all K splits or folds to derive a comprehensive performance metric for each model configuration.
5. **Model Selection**
    - **Optimal Model Identification:** Evaluate the aggregated performance of each model to select the one that demonstrates the best balance of accuracy, precision, recall, F1 score, or other relevant metrics, considering the specific objectives and constraints of the study.
6. **Final Model Training and Validation**
    - **Comprehensive Training:** Use the entire dataset (excluding any validation holdout) to train the selected model with the identified optimal hyperparameters.
    - **External Validation (Optional):** If a separate validation set was reserved or an external dataset is available, assess the finalized model against this data to gauge its performance and generalizability.
7. **Documentation and Transparency**
    - **In-depth Reporting:** Thoroughly document the selection process, including the rationale behind the choice of metrics, models, hyperparameters, and the comparative performance across different model configurations, to ensure clarity and reproducibility.

### **Additional Considerations**

- **Class Imbalances:** If applicable, ensure strategies to handle class imbalances (e.g., stratified sampling, class weights) are integrated into both the training and validation processes.
- **Computational Efficiency:** Be mindful of the computational complexity, particularly with a large number of models, hyperparameters, and folds. Employ efficient search techniques and parallel processing where feasible.
- **Domain Requirements:** Customize the model selection framework to align with the domain-specific needs and the nature of the data, ensuring the chosen approach is both relevant and practical.

## Simple k-fold

![[Notion/Getting into Data Science/Applied Data Science with Python/Classification, continued/Untitled.png|Untitled.png]]

## Nested k-fold

## Random k-fold

![[Notion/Getting into Data Science/Applied Data Science with Python/Classification, continued/Untitled 1.png|Untitled 1.png]]

![[Notion/Getting into Data Science/Applied Data Science with Python/Classification, continued/Untitled 2.png|Untitled 2.png]]

# Methods in detail

- **Data management**
    - `train_test_split` and `cross_val_score`
    - Rescaling inputs `StandardScaler` and `normalize`
    - Fill missing with `Imputer`
    - Split categorical into binary with `OneHotEncoder`
    - (_stretch_) Automated feature engineering (`SelectKBest`, `Recursive Feature Elimination` , auto feature eng)
- **Model details**
    - Explainability with feature weights, `SHAP`, `eli5`, and jackknife
    - Hyperparameter search (e.g., GridSearchCV)
- **Unsupervised learning**
    - Principal Component Analysis (PCA)
    - Clustering
    - K-nearest neighbor
    - t-distributed Stochastic Neighbor Embedding (t-SNE)
    - Self-organizing maps
- **Supervised classification**
    - Logistic regression
    - RandomForest
    - XGBoost
    - Neural networks (stretch goal)
    - LassoCV
- **Classical Statistics**
    
    1. **Hypothesis Testing:**
        - `**scipy.stats.ttest_ind**`: Performs a t-test for the means of two independent samples.
        - `**scipy.stats.f_oneway**`: Performs one-way ANOVA to test for differences between two or more groups.
        - `**scipy.stats.chi2_contingency**`: Conducts the chi-square test of independence for categorical variables.
        - **(Advanced)** `**statsmodels.stats.multicomp.pairwise_tukeyhsd**`: Performs Tukey's range test for multiple comparisons of means.
    2. **Regression Analysis:**
        1. Linear regression
        2. GLM
    
    ---
    
    Cut for prep time.
    
    1. **Time Series Analysis:**
        - `**statsmodels.tsa.arima.ARIMA**`: Fits an autoregressive integrated moving average (ARIMA) model for time series forecasting.
        - `**statsmodels.tsa.seasonal.seasonal_decompose**`: Decomposes a time series into trend, seasonal, and residual components for analysis.
    2. **Nonparametric Tests:**
        - `**scipy.stats.wilcoxon**`: Performs the Wilcoxon signed-rank test for comparing paired samples.
        - `**scipy.stats.mannwhitneyu**`: Performs the Mann-Whitney U test for comparing two independent samples.
        - `**scipy.stats.kruskal**`: Conducts the Kruskal-Wallis H test for comparing more than two independent samples.
    
    - **Sequential testing** and **CUPED** (stretch goal)

- Bayesian (stretch goal)
    1. **Bayesian T-Test:**
        - `**pymc3.stats.bayesian_ttest**`: This function from the PyMC3 library performs a Bayesian t-test, allowing for hypothesis testing with continuous data while considering uncertainty in the parameters. It can handle cases where the sample sizes are small or unequal.
    2. **Bayesian ANOVA:**
        - `**pymc3.stats.anova**`: PyMC3 also provides a Bayesian ANOVA implementation, allowing for hypothesis testing in the context of comparing means across multiple groups. It accounts for uncertainty in the parameters and provides posterior distributions for the effect sizes.
    3. **Bayesian Regression:**
        - `**pymc3.glm**`: The PyMC3 library offers tools for Bayesian generalized linear regression, which can be used for hypothesis testing in regression analysis. It allows for incorporating prior knowledge, handling uncertainty, and making probabilistic statements about regression coefficients.
    4. **Bayesian Significance Testing:**
        - `**BayesianSignificanceTest**`: This class from the `**bayesian-AB-testing**` library provides a Bayesian approach to A/B testing, allowing for hypothesis testing in the context of comparing two groups (e.g., treatment vs. control). It considers prior distributions for the parameters and updates beliefs based on observed data to assess significance.
    5. **Bayesian Sequential Testing:**
        - `**SequentialBayesianTest**`: You can use this class from the `**sequential-testing**` library to perform sequential hypothesis testing in a Bayesian framework. It allows for updating beliefs as data accumulates over time and making decisions based on posterior probabilities.

# Data preparation

## `**sklearn.model_selection.train_test_split**`

Takes two arrays of the same length and splits into train and test.

```Shell
import numpy as np
from sklearn.model_selection import train_test_split

# X = data, y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

Parameters:

- **test_size**, **train_size:** Float between 0..1, if either is not set then it receives the remainder, if both are not set the default is **train_size** = 0.25
- **random_state:** Shuffles predictively if set for reproducibility
- **shuffle:** Boolean on whether to shuffle or not
- **stratify:** Array of labels to stratify by, preserving class ratios. Default to None, [more details in docs](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)

## `**sklearn.model_selection.KFold**`

`**KFold**` is a method for splitting the dataset into **k consecutive folds** (without shuffling by default). Each fold is then used as a validation set once, while the k - 1 remaining folds form the training set.

```Python
from sklearn.model_selection import KFold

# Define the number of folds for cross-validation
num_folds = 5

# Create a KFold object
kf = KFold(n_splits=num_folds, shuffle=False, random_state=None)
```

Parameters:

- **n_splits:** int, default=5
    - Number of folds. Must be at least 2.
- **shuffle:** bool, default=False ⚠️
    - Whether to shuffle the data before splitting into batches.
- **random_state:** int or RandomState instance, default=None
    - When shuffle is True, random_state affects the ordering of the indices, which controls the randomness of each fold.
- **Other Parameters:** For more advanced usage, `**KFold**` also supports additional parameters like `**stratify**`, which enables stratified splitting based on labels.

## **Automated k-fold cross-validation with** `**cross_val_score**`

`**cross_val_score**` is a function provided by scikit-learn that enables users to perform k-fold cross-validation with ease. Here's an overview of how it works:

- **Description:** `**cross_val_score**` splits the dataset into multiple folds (or partitions) and evaluates the model's performance on each fold separately. It uses stratified K-fold cross-validation by default for classification tasks and K-fold cross-validation for regression tasks. The function returns an array of scores obtained for each fold, allowing users to assess the model's consistency across different subsets of the data.
- **Key Features:**
    - Model evaluation: `**cross_val_score**` provides a convenient way to evaluate the performance of a machine learning model using cross-validation, helping users estimate how well the model is likely to generalize to new data.
    - Consistency: By splitting the dataset into multiple folds and evaluating the model on each fold separately, `**cross_val_score**` provides more reliable performance estimates than single train-test splits, reducing the risk of overfitting and providing a more accurate assessment of model performance.
    - Customizability: `**cross_val_score**` allows users to specify the number of folds (`**cv**` parameter), the scoring metric (`**scoring**` parameter), and other relevant parameters to customize the cross-validation process according to their specific needs.
- **Parameters:**
    - `**estimator**`: The machine learning model or pipeline to evaluate.
    - `**X**`: The feature matrix (input data).
    - `**y**`: The target variable (labels) for supervised learning tasks.
    - `**cv**`: The number of folds for cross-validation. By default, it uses stratified K-fold cross-validation for classification tasks and K-fold cross-validation for regression tasks.
    - `**scoring**`: The scoring metric used to evaluate model performance (e.g., accuracy, precision, recall for classification tasks, and mean squared error, R-squared for regression tasks).
    - `**n_jobs**`: The number of CPU cores to use for parallelizing cross-validation. Set to `**1**` to use all available cores.
    - `**verbose**`: Controls the verbosity of cross-validation output (e.g., `**1**` for minimal output, `**2**` for more detailed output).
- **Example:**

```Python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the models
models = [LogisticRegression(), RandomForestClassifier()]

# Define the number of folds for cross-validation
num_folds = 5

# Create a cross-validation object with specific parameters
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation for each model
for model in models:
    # Perform cross-validation using the defined cross-validation object
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    # Print the cross-validation scores for the current model
    print(f"Model: {model.__class__.__name__}")
    print("Cross-Validation Scores:", scores)
    print("Mean Accuracy:", scores.mean())
```

## `sklearn.preprocessing.StandardScaler` and `normalize`

**Standardization** of datasets is a **common requirement for many machine learning estimators** implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with **zero mean and unit variance**

**Normalization** is the process of **scaling individual samples to have unit norm**. It requires specifying a normalization

  

```Shell
from sklearn.preprocessing import StandardScaler
# Scale the data
# When running without scaling the data, some models may not converge
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_01['image_flat'].tolist())
valid_scaled = scaler.transform(valid_01['image_flat'].tolist())

# Normalize a vector column
from sklearn.preprocessing import normalize
X['vector_normalized'] = normalize(X['vector'], norm='l2')
```

  

### When to rescale and which to use:

1. **Logistic Regression:**
    - **StandardScaler:** Logistic regression assumes that the numerical features are centered around zero with a standard deviation of one. Therefore, using `**StandardScaler**` is recommended to standardize the features before fitting the logistic regression model.
2. **Linear Regression:**
    - **StandardScaler:** Similar to logistic regression, linear regression assumes that the features are centered around zero with a standard deviation of one. Therefore, using `**StandardScaler**` is recommended for standardizing the features before fitting the linear regression model.
3. **Support Vector Machines (SVM):**
    - **StandardScaler:** SVMs are sensitive to the scale of the features. Standardizing the features using `**StandardScaler**` can help improve the performance and convergence of SVM models.
4. **k-Nearest Neighbors (KNN):**
    - **StandardScaler or normalize:** KNN calculates distances between data points, so the scale of the features directly affects the distance metric. Standardizing the features with `**StandardScaler**` or normalizing them with `**normalize**` can help ensure that all features contribute equally to the distance computation.
5. **Decision Trees and Random Forests:**
    - **No scaling required:** Decision trees and random forests are not sensitive to the scale of the features, so scaling is not necessary. You can use the raw features directly without applying any scaling.
6. **Gradient Boosting Machines (e.g., XGBoost, LightGBM):**
    - **No scaling required:** Gradient boosting algorithms like XGBoost and LightGBM are robust to feature scales and do not require scaling. You can use the raw features directly without applying any scaling.
7. **Neural Networks:**
    - **StandardScaler or normalize:** Neural networks can benefit from feature scaling to help improve convergence and training stability. Depending on the architecture and activation functions used in the neural network, you may choose to use `**StandardScaler**` or `**normalize**` to scale the features.

## `**sklearn.impute.SimpleImputer**`

`**SimpleImputer**` is a class in scikit-learn used to impute missing values in datasets. It replaces missing values with either a constant or statistics (mean, median, most frequent) along each column.

```Python
from sklearn.impute import SimpleImputer

# Create an instance of SimpleImputer with the desired strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the data
imputer.fit(X_train)

# Transform the data by imputing missing values
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

Parameters:

- **strategy:** Specifies the imputation strategy. It can be one of:
    - `**"mean"**`: Replace missing values using the mean along each column.
    - `**"median"**`: Replace missing values using the median along each column.
    - `**"most_frequent"**`: Replace missing values using the most frequent value along each column.
    - `**"constant"**`: Replace missing values with a constant specified by the `**fill_value**` parameter (default is 0).

### **When to use** `**SimpleImputer**`**:**

- **Preprocessing for various models:** `**SimpleImputer**` can be used as a preprocessing step before applying machine learning models. It handles missing values in the dataset, ensuring compatibility with models that cannot handle missing data.
- **Handling missing data in real-world datasets:** Real-world datasets often contain missing values due to various reasons such as data collection errors, incomplete records, or data corruption. `**SimpleImputer**` provides a convenient way to handle these missing values, allowing for reliable analysis and modeling.
- **Replacing missing values with appropriate statistics:** Depending on the nature of the data and the modeling task, different imputation strategies (e.g., mean, median, most frequent) may be more suitable. `**SimpleImputer**` allows for flexibility in choosing the appropriate imputation strategy based on the characteristics of the dataset.

## `**sklearn.preprocessing.OneHotEncoder**`

`**OneHotEncoder**` is a class in scikit-learn used to convert categorical integer features into one-hot encoded binary features. It creates a binary column for each category and returns a sparse matrix or a dense array depending on the `**sparse**` parameter.

```Python
from sklearn.preprocessing import OneHotEncoder

# Create an instance of OneHotEncoder
encoder = OneHotEncoder()

# Fit the encoder to the data
encoder.fit(X_train_categorical)

# Transform the data by one-hot encoding categorical features
X_train_encoded = encoder.transform(X_train_categorical)
X_test_encoded = encoder.transform(X_test_categorical)
```

Parameters:

- **sparse:** Specifies whether to return a sparse matrix (`**True**`) or a dense array (`**False**`). By default, it returns a sparse matrix.

### **When to use** `**OneHotEncoder**`**:**

- **Categorical features with non-ordinal values:** `**OneHotEncoder**` is suitable for encoding categorical features with non-ordinal values, where the categories do not have an inherent order.
- **Handling categorical data in machine learning models:** Many machine learning algorithms require numerical input data. `**OneHotEncoder**` allows you to convert categorical features into a format that can be used as input to these algorithms, improving model performance.
- **Avoiding numerical assumptions:** One-hot encoding prevents numerical algorithms from making assumptions about the ordinality or magnitude of categorical values, ensuring that each category is treated independently.
- **Categorical features with ordinal values:** Sometimes, categorical features may have ordinal values where the categories have a specific order or hierarchy. In such cases, using binning along with `**OneHotEncoder**` can be beneficial.
- **Handling numerical features with discrete intervals:** Binning is useful when dealing with numerical features that can be grouped into discrete intervals or categories. By binning numerical features and then applying `**OneHotEncoder**`, you can capture the underlying patterns or trends in the data more effectively.

## Feature engineering (_self-directed_)

In addition to the **Unsupervised methods** below, methods and libraries are available for feature engineering:

**Data reduction from existing features:**

- `SelectKBest`
- `Recursive Feature Elimination`

**Feature generation/combination**:

- `Featuretools` - automated feature engineering (create new features, keep best)

# Model details

## Feature explanation with `SHAP`, `eli5`, and jackknife

### `**SHAP**` (SHapley Additive exPlanations):

- **Description:** SHAP is a Python library for explaining the output of machine learning models using Shapley values, a concept from cooperative game theory. It provides both global and local explanations for model predictions, allowing users to understand the impact of individual features on model outcomes.
- **Use Case:** When you need both global and local explanations for model predictions.
- **Key Features:**
    - Model-agnostic: SHAP is compatible with a wide range of machine learning models, including tree-based models, linear models, neural networks, and more.
    - Global and local explanations: SHAP enables users to obtain both global feature importance measures and local explanations for individual predictions, providing insights into model behavior at different levels.
    - Consistency: SHAP ensures consistent explanations across different machine learning models, making it suitable for comparing and interpreting models from various domains.
    - Visualization: SHAP provides visualizations such as summary plots, force plots, and dependence plots to help users interpret and communicate model explanations effectively.

### `**eli5**` (Explain Like I'm 5):

- **Description:** `**eli5**` is a Python library for debugging and explaining machine learning models. It offers tools to inspect model coefficients, feature importances, and predictions, making it easier to understand model behavior and identify potential issues.
- **Use Case:** When you need human-readable explanations for model predictions and feature importance.
- **Key Features:**
    - Model-agnostic: `**eli5**` supports various machine learning models and frameworks, making it versatile for explaining models across different domains.
    - Interpretability: `**eli5**` provides human-readable explanations for model predictions, making complex models more interpretable and accessible to users.
    - Compatibility: `**eli5**` integrates seamlessly with popular machine learning libraries such as scikit-learn, XGBoost, LightGBM, and more, allowing for easy integration into existing workflows.
    - Transparency: `**eli5**` offers transparent and intuitive explanations for model behavior, helping users gain insights into the underlying factors driving model predictions.

### Jackknife:

- **Description:** Jackknife is a resampling technique commonly used in statistics for estimating the precision of parameter estimates and assessing the stability of model predictions. It involves **systematically leaving out one column at a time from the dataset** and recalculating model predictions to evaluate the impact of individual data points on model performance.
- **Use Case:** When you need to assess the stability and reliability of model predictions.
- **Key Features:**
    - Robustness: Jackknife provides a robust method for assessing the stability and reliability of machine learning models by systematically evaluating model performance across different subsets of the data.
    - Bias correction: Jackknife can be used to estimate and correct for bias in parameter estimates and predictions, improving the accuracy and generalization of machine learning models.
    - Versatility: Jackknife is a versatile technique that can be applied to various machine learning tasks, including regression, classification, and model evaluation, making it suitable for a wide range of applications.

## H**yperparameter Search**

Hyperparameter search is a critical step in optimizing machine learning models for performance and generalization. Several techniques and libraries are available for efficiently searching and tuning hyperparameters to improve model accuracy and robustness. Here are some commonly used approaches:

### Grid Search (most common): `**sklearn.model_selection.GridSearchCV**`

- **Description:** Grid search is a hyperparameter optimization technique that exhaustively searches through a specified grid of hyperparameter values to identify the combination that yields the best model performance. It evaluates the model's performance for each combination of hyperparameters using cross-validation and selects the combination with the highest validation score.
- **Key Features:**
    - Exhaustive search: Grid search systematically evaluates all possible combinations of hyperparameters within the specified grid, ensuring thorough exploration of the hyperparameter space.
    - Cross-validation: Grid search typically employs cross-validation to estimate the generalization performance of each hyperparameter combination, reducing the risk of overfitting and providing more reliable performance estimates.
    - Transparency: Grid search is transparent and easy to understand, making it suitable for users who prefer a straightforward approach to hyperparameter tuning.
- **Parameters:**
    - `**param_grid**`: Dictionary or list of dictionaries specifying the hyperparameter grid to search.
    - `**scoring**`: The scoring metric used to evaluate the model performance (e.g., accuracy, precision, recall).
    - `**cv**`: The number of folds for cross-validation.
    - `**n_jobs**`: The number of CPU cores to use for parallelizing the grid search. Set to `**1**` to use all available cores.
    - `**verbose**`: Controls the verbosity of grid search output (e.g., `**1**` for minimal output, `**2**` for more detailed output).

```Python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Instantiate the Random Forest classifier
rf = RandomForestClassifier()

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

### Random Search: `**sklearn.model_selection.RandomizedSearchCV**`

- **Description:** Random search is a hyperparameter optimization technique that randomly samples hyperparameter values from specified distributions and evaluates the model's performance for each sampled combination. It aims to efficiently explore the hyperparameter space by focusing on promising regions while avoiding exhaustive search.
- **Key Features:**
    - Efficiency: Random search can be more efficient than grid search for high-dimensional hyperparameter spaces, as it does not require evaluating every possible combination of hyperparameters.
    - Exploration-exploitation trade-off: Random search strikes a balance between exploration (sampling diverse hyperparameter values) and exploitation (selecting promising regions based on previous evaluations), allowing for more effective hyperparameter tuning.
    - Flexibility: Random search offers flexibility in defining the search space and sampling distributions for each hyperparameter, making it suitable for a wide range of machine learning tasks and algorithms.
- **Parameters:**
    - Includes all from `**GridSearchCV**`
    - `**n_iter**`: The number of parameter settings that are sampled.

```Python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import randint

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the parameter distributions
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 10)
}

# Instantiate the Random Forest classifier
rf = RandomForestClassifier()

# Perform random search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
random_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

### **Bayesian Optimization:** `**scikit-optimize**` **or** `**BayesianOptimization**`

- **Description:** Bayesian optimization is a sequential model-based optimization technique that uses probabilistic models to guide the search for optimal hyperparameters. It models the objective function (model performance) as a probability distribution and iteratively selects hyperparameters to maximize an acquisition function that balances exploration and exploitation.
- **Use Case:** When you want to automate the hyperparameter tuning process and efficiently search for optimal hyperparameters.
- **Key Features:**
    - Sequential optimization: Bayesian optimization iteratively selects hyperparameters based on previous evaluations, gradually refining the search space and focusing on promising regions to improve optimization efficiency.
    - Probabilistic modeling: Bayesian optimization models the objective function as a probability distribution, allowing it to make informed decisions about which hyperparameters to explore next.
    - Efficient exploration: Bayesian optimization uses probabilistic surrogate models to estimate the objective function, allowing it to explore the hyperparameter space more efficiently than exhaustive search methods like grid search.
- **Libraries:**
    - `**scikit-optimize**` **(skopt)**: This library provides a simple and efficient implementation of Bayesian optimization techniques for hyperparameter tuning. It offers various optimization algorithms, including Gaussian process-based models, and supports both continuous and categorical hyperparameters.
    - `**BayesianOptimization**`: Another popular library for Bayesian optimization, which allows you to define your objective function and search space easily. It offers a user-friendly interface for conducting Bayesian optimization experiments and supports parallel optimization for improved efficiency.
- **Parameters (for** `**scikit-optimize**`**):**
    - `**space**`: The search space defining the range or distribution of each hyperparameter.
    - `**acq_func**`: The acquisition function used to guide the search process (e.g., expected improvement, probability of improvement).
    - `**n_calls**`: The number of function evaluations (model training iterations) to perform during optimization.
    - `**n_initial_points**`: The number of initial random evaluations to bootstrap the optimization process.
    - `**n_jobs**`: The number of parallel jobs to run during optimization for improved efficiency.
    - `**random_state**`: Seed for random number generation to ensure reproducibility.
- **Example (using** `**scikit-optimize**`**):**

```Python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the search space
param_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(2, 20),
    'min_samples_split': Integer(2, 10)
}

# Instantiate the Random Forest classifier
rf = RandomForestClassifier()

# Perform Bayesian optimization
bayes_search = BayesSearchCV(estimator=rf, search_spaces=param_space, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1)
bayes_search.fit(X, y)

# Print the best parameters and best score
print("Best Parameters:", bayes_search.best_params_)
print("Best Score:", bayes_search.best_score_)

```

In this example, we use `**BayesSearchCV**` from `**scikit-optimize**` to perform Bayesian optimization for hyperparameter tuning of a Random Forest classifier on the Iris dataset. We define the search space for hyperparameters using `**Integer**`, `**Real**`, or `**Categorical**` types from `**skopt.space**`, specify the number of iterations (`**n_iter**`), and other relevant parameters. Finally, we fit the `**BayesSearchCV**` object to the data and print the best parameters and best score obtained during the search.

# Unsupervised learning

## Principal Component Analysis `**sklearn.decomposition.PCA**`

**PCA (Principal Component Analysis)** is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving most of the variability in the data.

```Python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the transformed data
plt.figure(figsize=(8, 6))
for target in set(y):
    plt.scatter(X_pca[y == target, 0], X_pca[y == target, 1], label=f'Class {target}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()
```

### **Parameters:**

- `**n_components**`: int, float, None or str
    - Number of components to keep. If `**n_components**` is not set, all components are kept (min(n_samples, n_features)).
    - If `**n_components**` is an int, it specifies the number of principal components to retain.
    - If `**n_components**` is a float between 0.0 and 1.0, it indicates the ratio of variance to preserve.
    - If `**n_components**` is 'mle', it uses Minka’s MLE to guess the dimension.
    - If `**n_components**` is None, it keeps all components.

### **Outputs:**

- `**components_**`: array-like, shape (n_components, n_features)
    - Principal axes in feature space, representing the directions of maximum variance in the data.
- `**explained_variance_**`: array-like, shape (n_components,)
    - The amount of variance explained by each of the selected components.
- `**explained_variance_ratio_**`: array-like, shape (n_components,)
    - Percentage of variance explained by each of the selected components.

### **When to use** `**PCA**`**:**

- **Dimensionality reduction:** PCA is commonly used for reducing the dimensionality of high-dimensional data while retaining most of the variability. It is useful for visualizing high-dimensional data, speeding up subsequent computations, and reducing the risk of overfitting in models.
- **Visualization:** PCA can be used to visualize high-dimensional data by projecting it onto a lower-dimensional space (e.g., 2D or 3D). This allows for easier interpretation and visualization of the data's structure and relationships.
- **Noise reduction:** PCA can help in reducing the effects of noise and irrelevant features in the data by focusing on the directions of maximum variance. This can lead to improved model performance and generalization.
- **Feature engineering:** PCA can be used as a feature engineering technique to create new features that capture the most important patterns in the data. These new features can then be used as inputs to machine learning models for improved performance.

## **KMeans Clustering** `**sklearn.cluster.KMeans**`

**KMeans Clustering** is an unsupervised learning algorithm used to partition a dataset into clusters by minimizing the within-cluster variance. It iteratively assigns data points to the nearest cluster centroid and updates the centroids based on the mean of the points in each cluster.

```Python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Plot the clustered data
plt.figure(figsize=(8, 6))
for cluster in set(labels):
    plt.scatter(X[labels == cluster, 0], X[labels == cluster, 1], label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering of Iris Dataset')
plt.legend()
plt.show()
```

### **Parameters:**

- `**n_clusters**`: int
    - The number of clusters to form as well as the number of centroids to generate.

### **Outputs:**

- `**labels_**`: array-like, shape (n_samples,)
    - Cluster labels for each data point.

### **When to use** `**KMeans**`**:**

- **Cluster analysis:** KMeans is commonly used for clustering analysis to discover natural groupings or patterns within data.
- **Segmentation:** KMeans can be used for customer segmentation, market segmentation, and image segmentation, among other applications.
- **Anomaly detection:** KMeans can help identify outliers or anomalies by assigning data points to clusters, and data points that are distant from cluster centroids may be considered as anomalies.
- **Feature engineering:** KMeans can be used as a feature engineering technique to create new features based on cluster assignments, which can then be used as input features for machine learning models.

## **K-nearest Neighbors** `**sklearn.neighbors.KNeighborsClassifier**`

**K-nearest Neighbors (KNN)** is a non-parametric classification algorithm that classifies data points based on the majority class among their k nearest neighbors. It is widely used for both classification and regression tasks.

```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# We'll use mlxtend to make the plotting easier
from mlxtend.plotting import plot_decision_regions

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Considering only the first two features for visualization
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Plot decision regions
plot_decision_regions(X_train, y_train, clf=knn, legend=2)

# Adding axes annotations
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Decision Boundaries (k=3)')
plt.show()
```

### **Parameters:**

- `**n_neighbors**`: int
    - Number of neighbors to use for classification.

### **Outputs:**

- **Predictions**: array-like, shape (n_samples,)
    - Predicted class labels for each data point in the test set.

### **When to use** `**KNeighborsClassifier**`**:**

- **Classification tasks:** KNN is primarily used for classification tasks where the decision boundary is not linear.
- **Small to medium-sized datasets:** KNN performs well on small to medium-sized datasets with a relatively low number of features. It can be computationally expensive for large datasets due to the need to compute distances between data points.
- **Non-parametric learning:** KNN is a non-parametric learning algorithm, meaning it does not make any assumptions about the underlying data distribution. It can capture complex relationships between features and target variables.
- **No training phase:** KNN does not have a training phase, as it memorizes the training data and makes predictions based on the entire dataset. This makes it suitable for online learning scenarios where new data points are continuously added to the dataset.

## **t-SNE** `**sklearn.manifold.TSNE**`

**t-distributed Stochastic Neighbor Embedding (t-SNE)** is a dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space. It aims to preserve the local structure of the data by modeling pairwise similarities between data points in high-dimensional space and low-dimensional space.

```Python
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Plot the transformed data
plt.figure(figsize=(8, 6))
for target in set(y):
    plt.scatter(X_tsne[y == target, 0], X_tsne[y == target, 1], label=f'Class {target}')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Iris Dataset')
plt.legend()
plt.show()

```

### **Parameters:**

- `**n_components**`: int
    - Dimension of the embedded space (default is 2).

### **Outputs:**

- **Embedded data**: array-like, shape (n_samples, n_components)
    - Embedded data points in the lower-dimensional space.

### **When to use** `**TSNE**`**:**

- **Visualization:** t-SNE is primarily used for visualizing high-dimensional data in a lower-dimensional space (typically 2D or 3D). It preserves the local structure of the data, making it suitable for exploring and interpreting complex datasets.
- **Exploratory data analysis:** t-SNE can be used as an exploratory data analysis tool to gain insights into the underlying structure and relationships within the data. It helps identify clusters, patterns, and outliers that may not be apparent in the original high-dimensional space.
- **Feature engineering:** t-SNE can be used as a feature engineering technique to create new features based on the low-dimensional representations of the data. These new features can then be used as input features for machine learning models, potentially improving model performance.

## **Self-Organizing Maps (SOM)**

Self-Organizing Maps (SOM) is an unsupervised learning algorithm used for dimensionality reduction and visualization of high-dimensional data. It is a type of artificial neural network that learns to map high-dimensional input data onto a low-dimensional grid of neurons, preserving the topological relationships between data points.

### **Libraries available:**

1. **Minisom**: A minimalistic and efficient implementation of Self-Organizing Maps (SOM) in Python.
2. **SOMPY**: A Python library for self-organizing maps that supports both batch and online training.

```Python
from minisom import MiniSom
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the SOM
som = MiniSom(7, 7, X.shape[1], sigma=0.5, learning_rate=0.5)

# Train the SOM
som.train(X, 100)

# Visualize the SOM
plt.figure(figsize=(8, 6))
for i, (x, label) in enumerate(zip(X, y)):
    winner = som.winner(x)
    plt.text(winner[0]+.5,  winner[1]+.5,  str(label), color=plt.cm.tab10(label / 10.), fontdict={'weight': 'bold', 'size': 11})
plt.title('Self-Organizing Map of Iris Dataset')
plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
plt.colorbar()
plt.show()

```

### **Parameters:**

- `**x**`: int
    - Number of neurons in the x-axis (width) of the SOM grid.
- `**y**`: int
    - Number of neurons in the y-axis (height) of the SOM grid.
- `**input_len**`: int
    - Number of features in the input data.
- `**sigma**`: float
    - Spread of the neighborhood function (default is 1.0).
- `**learning_rate**`: float
    - Initial learning rate (default is 0.5).

### **Outputs:**

- `**distance_map**`: array-like, shape (x, y)
    - Distance map of the SOM grid, indicating the average distance between each neuron and its neighbors.
- `**win_map**`: dictionary
    - Mapping between each input data point and its corresponding winning neuron on the SOM grid.

### **When to use SOM:**

- **Dimensionality reduction:** SOM can be used for dimensionality reduction of high-dimensional data, projecting it onto a low-dimensional grid while preserving the topological relationships between data points.
- **Visualization:** SOM provides a visual representation of the high-dimensional data in a lower-dimensional space, making it easier to explore and interpret complex datasets.
- **Clustering:** SOM can be used for clustering similar data points together on the SOM grid, helping identify clusters and patterns in the data.
- **Feature extraction:** SOM can extract meaningful features from the input data, which can be used as input features for downstream machine learning models.

### Example in the wild:

I used `**minisom**` for feature engineering in this project:

https://github.com/christopherseaman/five_twelve

# Supervised classification

## **Logistic Regression** `**sklearn.linear_model.LogisticRegression**`

**Logistic Regression** is a supervised learning algorithm used for binary classification tasks. It models the probability that a given input belongs to a particular class using the logistic function.

```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features for visualization
y = iris.target

# Apply Logistic Regression for classification
logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logistic_regression.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

### **Parameters:**

- `**penalty**`: str, {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
    - The penalty (regularization term) to be used. Default is 'l2'.
- `**C**`: float, default=1.0
    - Inverse of regularization strength; smaller values specify stronger regularization.
- `**solver**`: {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default='lbfgs'
    - Algorithm to use in the optimization problem.
- `**max_iter**`: int, default=100
    - Maximum number of iterations taken for the solvers to converge.

### **Outputs:**

- `**coef_**`: array-like, shape (1, n_features) or (n_classes, n_features)
    - Coefficients of the features in the decision function.
- `**intercept_**`: array-like, shape (1,) or (n_classes,)
    - Intercept (a.k.a. bias) added to the decision function.

### **When to use** `**Logistic Regression**`**:**

- **Binary classification:** Logistic Regression is well-suited for binary classification tasks where the target variable has two possible classes.
- **Interpretability:** Logistic Regression provides interpretable results, as the coefficients of the model indicate the strength and direction of the relationships between features and the target variable.
- **Linear decision boundaries:** Logistic Regression assumes a linear relationship between the features and the log-odds of the target variable, resulting in linear decision boundaries between classes.
- **Scalability:** Logistic Regression can handle large datasets efficiently, making it suitable for scenarios with a large number of samples or features.

  

## **Random Forest** `**sklearn.ensemble.RandomForestClassifier**`

**Random Forest** is an ensemble learning method that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. It can be used for classification and regression tasks.

```Python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply Random Forest Classifier for classification
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.bar(range(X.shape[1]), random_forest.feature_importances_)
plt.xticks(range(X.shape[1]), iris.feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.show()
```

### **Parameters:**

- `**n_estimators**`: int, default=100
    - The number of trees in the forest.
- `**criterion**`: {"gini", "entropy"}, default="gini"
    - The function to measure the quality of a split.
- `**max_depth**`: int, default=None
    - The maximum depth of the tree.
- `**min_samples_split**`: int or float, default=2
    - The minimum number of samples required to split an internal node.
- `**min_samples_leaf**`: int or float, default=1
    - The minimum number of samples required to be at a leaf node.
- `**max_features**`: {"auto", "sqrt", "log2"}, int, float, or None, default="auto"
    - The number of features to consider when looking for the best split.

### **Outputs:**

- `**feature_importances_**`: array-like of shape (n_features,)
    - The feature importances (the higher, the more important the feature).

### **When to use** `**Random Forest**`**:**

- **Classification and regression:** Random Forest can be used for both classification and regression tasks, making it a versatile algorithm suitable for various types of predictive modeling problems.
- **High-dimensional data:** Random Forest is effective for datasets with a large number of features and can handle both numerical and categorical variables without the need for feature scaling.
- **Feature importance:** Random Forest provides feature importances, allowing for insight into the most influential features in the dataset, which can aid in feature selection and interpretation.
- **Robustness to overfitting:** Random Forest mitigates overfitting by averaging multiple decision trees trained on different subsets of the data, leading to more robust and generalizable models.

## **XGBoost Classifier** `**xgboost.XGBClassifier**`

**XGBoost (Extreme Gradient Boosting)** is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

```Python
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply XGBoost Classifier for classification
xgb_classifier = XGBClassifier(n_estimators=100)
xgb_classifier.fit(X, y)

# Plot feature importances
plt.figure(figsize=(8, 6))
plt.bar(range(X.shape[1]), xgb_classifier.feature_importances_)
plt.xticks(range(X.shape[1]), iris.feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('XGBoost Feature Importances')
plt.show()
```

### **Parameters:**

- `**n_estimators**`: int, default=100
    - Number of gradient boosted trees (or boosting rounds).
- `**learning_rate**`: float, default=0.1
    - Step size shrinkage used in update to prevent overfitting. Range is [0,1].
- `**max_depth**`: int, default=6
    - Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
- `**subsample**`: float, default=1
    - Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost randomly samples half of the training data prior to growing trees.
- `**colsample_bytree**`: float, default=1
    - Subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.

### **Outputs:**

- `**feature_importances_**`: array-like of shape (n_features,)
    - The feature importances (the higher, the more important the feature).

### **When to use** `**XGBoost**`**:**

- **Classification and regression:** XGBoost is a powerful algorithm suitable for both classification and regression tasks. It is widely used in machine learning competitions and real-world applications due to its high predictive performance.
- **Highly scalable:** XGBoost is highly efficient and scalable, making it suitable for large datasets with a high number of features and samples. It can be parallelized and distributed across multiple CPU cores and GPUs for faster training.
- **Handling missing values:** XGBoost can handle missing values internally, eliminating the need for imputation or preprocessing steps. It also supports sparse data formats, making it efficient for datasets with a large number of missing values or features.
- **Feature importance:** XGBoost provides feature importances, allowing for insight into the most influential features in the dataset. This can aid in feature selection, interpretation, and understanding of the underlying patterns in the data.

## Understanding the types of trees

### **Decision Trees:**

- **Concept:** Decision trees are hierarchical structures that recursively split the dataset into subsets based on the values of input features. Each internal node represents a feature and a decision based on its value, while each leaf node represents the outcome or prediction.
- **Splitting Criteria:** Decision trees split the data at each node based on a criterion that maximizes the homogeneity of the target variable within the resulting subsets. Common splitting criteria include Gini impurity and information gain (entropy).
- **Tree Growth:** Decision trees can continue growing until certain stopping criteria are met, such as reaching a maximum depth, minimum number of samples per leaf, or a minimum impurity threshold.
- **Predictions:** To make predictions for a new data point, it traverses the tree from the root node to a leaf node based on the feature values of the data point. The prediction at the leaf node is typically the majority class (for classification) or the average value (for regression) of the training samples in that node.

### **Random Forests:**

- **Ensemble Learning:** Random forests are an ensemble learning method that consists of multiple decision trees trained on different subsets of the data. Each tree is trained independently, typically using a random subset of features at each node.
- **Voting:** During prediction, each tree in the random forest independently predicts the outcome, and the final prediction is determined by a majority vote (for classification) or averaging (for regression) of the individual tree predictions.
- **Reducing Overfitting:** Random forests mitigate overfitting by averaging the predictions of multiple trees, reducing the risk of memorizing noise in the training data.

### **XGBoost:**

- **Gradient Boosting:** XGBoost is an implementation of gradient boosting, a sequential ensemble learning technique. Unlike random forests, where trees are built independently, in gradient boosting, trees are built sequentially, with each new tree aiming to correct the errors made by the previous ones.
- **Iterative Improvement:** XGBoost builds trees in an additive manner, where each new tree is trained to minimize the errors (residuals) of the combined predictions of the existing trees.
- **Regularization:** XGBoost incorporates regularization techniques to control model complexity and prevent overfitting, such as shrinkage (learning rate) and tree depth regularization.

### **Key Differences:**

1. **Independence vs. Collaboration:** In random forests, each tree makes its own prediction independently, while in XGBoost, the trees work together collaboratively to improve predictions over time.
2. **Bias-Variance Tradeoff:** Random forests tend to have low bias and high variance, while XGBoost aims to reduce both bias and variance simultaneously, often resulting in better overall performance.
3. **Interpretability:** Random forests are easier to interpret since each tree can be analyzed individually, while XGBoost's ensemble approach may be harder to interpret due to the collaborative nature of the trees.

## Keras Neural Network (multi-layer perceptron)

> [!important]  
> PREVIEW: We will discuss neural networks in more detail next lecture  

**Keras** is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, Theano, or CNTK. It allows for easy and fast experimentation with deep neural networks.

### **What is a Multi-Layer Perceptron (MLP)?**

A Multi-Layer Perceptron (MLP) is a type of feedforward neural network consisting of multiple layers of nodes, or neurons. Each neuron in one layer connects with a certain weight to every neuron in the next layer. MLPs are often used for supervised learning tasks such as classification and regression.

```Python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values to [0, 1]

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D array
    Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    Dropout(0.2),                    # Dropout layer with a dropout rate of 20%
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes and softmax activation
])

# Compile the model
model.compile(optimizer='adam',                       # Adam optimizer
              loss='sparse_categorical_crossentropy', # Sparse categorical crossentropy loss
              metrics=['accuracy'])                   # Accuracy metric for evaluation

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
```

### **Model Architecture (3 layers):**

- **Flatten Layer:** Converts the 2D input data (28x28 images) into a 1D array.
- **Dense Layer (Hidden Layer):** Fully connected neural network layer with 128 units and ReLU activation function.
- **Dense Layer (Output Layer):** Fully connected layer with 10 units corresponding to the 10 classes in the MNIST dataset, using softmax activation for multi-class classification.

### **Altering Neural Networks for Different Classification Tasks**

1. **Multi-Class Classification (e.g., 26 letters):**
    
    - Adjust the number of neurons in the output layer to match the number of classes. For example, if you're classifying among 26 letters, the output layer should have 26 neurons.
    - Utilize the softmax activation function in the output layer for multi-class classification. It provides probabilities for each class and ensures that the sum of the probabilities across all classes is equal to 1.
    - Use appropriate loss functions such as categorical cross-entropy for multi-class classification tasks.
    
    ```Python
    # Define the model architecture for multi-class classification
    model_multi_class = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D array
        Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
        Dense(26, activation='softmax')  # Output layer with 26 neurons for 26 classes and softmax activation
    ])
    
    # Compile the model for multi-class classification
    model_multi_class.compile(optimizer='adam',                       # Adam optimizer
                               loss=SparseCategoricalCrossentropy(),  # Sparse categorical crossentropy loss
                               metrics=['accuracy'])                  # Accuracy metric for evaluation
    ```
    
2. **Binary Classification (e.g., binary classifier):**
    
    - Modify the number of neurons in the output layer to 1, representing the probability of belonging to the positive class (e.g., 1 for positive class, 0 for negative class).
    - Use the sigmoid activation function in the output layer for binary classification. It squashes the output between 0 and 1, representing the probability of belonging to the positive class.
    - `BinaryCrossentropy` loss function is commonly used for binary classification tasks.
    
    ```Python
    # Define the model architecture for binary classification
    model_binary_class = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D array
        Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
        Dense(1, activation=sigmoid)    # Output layer with 1 neuron for binary classification and sigmoid activation
    ])
    
    # Compile the model for binary classification
    model_binary_class.compile(optimizer='adam',                 # Adam optimizer
                               loss=BinaryCrossentropy(),       # Binary crossentropy loss
                               metrics=['accuracy'])            # Accuracy metric for evaluation
    ```
    

### **Training Parameters:**

- `**optimizer**`: Adam optimizer.
- `**loss**`: Categorical cross-entropy loss function.
- `**metrics**`: Accuracy is used as the evaluation metric.

### **Outputs:**

- **Test Accuracy:** The accuracy achieved by the model on the test dataset.

### **When to use Keras Neural Networks:**

- **Image classification:** Keras is suitable for building neural network models for image classification tasks, such as MNIST digit classification.
- **Fast prototyping:** Keras allows for easy and fast experimentation with different neural network architectures, making it ideal for rapid prototyping of deep learning models.
- **Integration with TensorFlow:** Keras seamlessly integrates with TensorFlow, allowing for efficient computation on both CPU and GPU architectures.
- **Community support:** Keras has a large and active community, with extensive documentation, tutorials, and pre-trained models available for various tasks.

## Bonus methods

- **Lasso Regression** `**sklearn.linear_model.Lasso**` **:**
- **Lasso Cross-Validation** `**sklearn.linear_model.LassoCV**` **:**

# Hypothesis testing

## _**t**_**-test** `**scipy.stats.ttest_ind**`

`**scipy.stats.ttest_ind**` is a function in the `**scipy.stats**` module used to perform an independent two-sample t-test for comparing the means of two independent samples.

```Python
from scipy import stats

# Example usage of ttest_ind
sample1 = [1, 2, 3, 4, 5]
sample2 = [6, 7, 8, 9, 10]
t_statistic, p_value = stats.ttest_ind(sample1, sample2)
print("t-statistic:", t_statistic)
print("p-value:", p_value)
```

### **Parameters:**

- `**a, b**`: array_like
    - The arrays containing the sample data for which the t-test is to be performed.
- `**equal_var**`: bool, optional
    - If `**True**` (default), perform a standard independent two-sample t-test that assumes equal population variances. If `**False**`, perform Welch’s t-test, which does not assume equal population variances.

### **Outputs:**

- `**t_statistic**`: float
    - The calculated t-statistic for the two samples.
- `**p_value**`: float
    - The two-tailed p-value associated with the t-statistic.

### **When to use** `**scipy.stats.ttest_ind**`**:**

- **Comparing means of two samples:** Use the independent t-test when you want to determine whether there is a statistically significant difference between the means of two independent samples.
- **Normality assumptions:** The t-test assumes that the data in each sample are approximately normally distributed, with equal or at least approximately equal variances between groups.
- **Independent samples:** The samples should be independent of each other, meaning that the observations in one sample are not related to the observations in the other sample.

## **ANOVA** `**scipy.stats.f_oneway**`

`**scipy.stats.f_oneway**` is a function in the `**scipy.stats**` module used to perform a one-way ANOVA (Analysis of Variance) test to compare the means of two or more independent samples.

```Python
from scipy import stats

# Example usage of f_oneway
sample1 = [1, 2, 3, 4, 5]
sample2 = [6, 7, 8, 9, 10]
f_statistic, p_value = stats.f_oneway(sample1, sample2)
print("F-statistic:", f_statistic)
print("p-value:", p_value)

```

### **Parameters:**

- `**sample1, sample2, ...**`: array_like
    - The arrays containing the sample data for which the one-way ANOVA test is to be performed.

### **Outputs:**

- `**f_statistic**`: float
    - The calculated F-statistic for the samples.
- `**p_value**`: float
    - The p-value associated with the F-statistic.

### **When to use** `**scipy.stats.f_oneway**`**:**

- **Comparing means of multiple samples:** Use the one-way ANOVA test when you want to determine whether there are statistically significant differences between the means of two or more independent samples.
- **Normality assumptions:** The ANOVA test assumes that the data in each sample are approximately normally distributed.
- **Equal variances:** The ANOVA test assumes that the variances of the populations from which the samples are drawn are approximately equal.

## **Chi^2** `**scipy.stats.chi2_contingency**`

`**scipy.stats.chi2_contingency**` is a function in the `**scipy.stats**` module used to perform a chi-square test of independence for contingency tables.

```Python
from scipy import stats

# Example usage of chi2_contingency
observed = [[10, 20, 30], [40, 50, 60]]
chi2_statistic, p_value, dof, expected = stats.chi2_contingency(observed)
print("Chi-square statistic:", chi2_statistic)
print("p-value:", p_value)
print("Degrees of freedom:", dof)
print("Expected frequencies:", expected)

```

### **Parameters:**

- `**observed**`: array_like
    - The observed frequencies in the contingency table. The table should have at least two rows and two columns.

### **Outputs:**

- `**chi2_statistic**`: float
    - The calculated chi-square statistic.
- `**p_value**`: float
    - The p-value associated with the chi-square statistic.
- `**dof**`: int
    - The degrees of freedom.
- `**expected**`: ndarray
    - The expected frequencies, based on the marginal sums of the contingency table.

### **When to use** `**scipy.stats.chi2_contingency**`**:**

- **Testing independence of categorical variables:** Use the chi-square test of independence when you want to determine whether there is a statistically significant association between two or more categorical variables.
- **Contingency tables:** The variables should be measured on a nominal scale, and the data should be arranged in a contingency table format.
- **Assumptions:** The chi-square test assumes that the observations are independent and that the expected frequencies are not too small (typically, all expected frequencies should be greater than 5).

# Regression

## `**sklearn.linear_model.LinearRegression**`

**Linear Regression** is a supervised learning algorithm used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

```Python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the California housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on new data
y_pred = model.predict(X_test)
```

### **Parameters:**

- `**fit_intercept**`: bool, default=True
    - Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g., data is expected to be centered).
- `**normalize**`: bool, default=False
    - This parameter is ignored when `**fit_intercept**` is set to False. If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use `**sklearn.preprocessing.StandardScaler**` before calling `**fit**` on an estimator with `**normalize=False**`.
- `**copy_X**`: bool, default=True
    - If True, X will be copied; else, it may be overwritten.
- `**n_jobs**`: int, default=None
    - The number of jobs to use for the computation. This will only provide speedup for n_targets > 1 and sufficient large problems.

### **Attributes (learned during training):**

- `**coef_**`: array, shape (n_features, ) or (n_targets, n_features)
    - Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.
- `**intercept_**`: array
    - Independent term in the linear model.

### **Outputs (from completed model):**

- **Predictions (**`**y_pred**`**)**: array
    - The predicted values of the dependent variable based on the fitted linear model.
- **Residuals**: array
    - The differences between the observed and predicted values of the dependent variable. These residuals can be used to assess the goodness of fit of the model.

### **When to use** `**LinearRegression**`**:**

- **Simple and interpretable:** Linear regression is a simple and interpretable method for modeling the relationship between independent and dependent variables. It provides insight into the impact of each independent variable on the dependent variable.
- **Linear relationship:** Linear regression assumes a linear relationship between the independent and dependent variables. It is suitable when the relationship between variables can be reasonably approximated by a straight line.
- **Assumption of normality:** Linear regression assumes that the residuals (the differences between observed and predicted values) are normally distributed. It is important to check this assumption when using linear regression for inference.
- **Predictive modeling:** Linear regression can also be used for predictive modeling when the goal is to predict the value of the dependent variable based on the values of the independent variables.

## **Generalized Linear Models** `**statsmodels.api.GLM**`

**Generalized Linear Models (GLMs)** extend the linear regression framework to accommodate different types of response variables and error distributions. They are particularly useful when the response variable is not normally distributed or when the relationship between the independent variables and the mean of the response variable is not linear.

GLMs are particularly useful when the response variable is not normally distributed or when the relationship between the independent variables and the mean of the response variable is not linear. GLMs allow for the specification of a link function and an error distribution, which can be tailored to the specific characteristics of the data.

Common examples of GLMs include:

1. **Logistic Regression**: Used for binary classification problems where the response variable is binary (0 or 1).
2. **Poisson Regression**: Used for count data where the response variable represents the number of occurrences of an event in a fixed interval of time or space.
3. **Gamma Regression**: Used for continuous, positive-valued response variables that are skewed and do not follow a normal distribution.
4. **Inverse Gaussian Regression**: Used for continuous, positive-valued response variables that are skewed and have a long right tail.

```Python
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

# Load the California housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Add a constant for the intercept
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a GLM
glm_model = sm.GLM(y_train, X_train, family=sm.families.Gaussian())
glm_results = glm_model.fit()

# Print the summary of the model
print(glm_results.summary())
```

### **Parameters:**

- `**endog**`: array_like
    - The response variable.
- `**exog**`: array_like
    - The design matrix.
- `**family**`: family class instance
    - The distribution family of the GLM.

### **Attributes (learned during training):**

- `**params**`: array
    - Estimated coefficients for the GLM.
- `**bse**`: array
    - Standard errors of the coefficient estimates.
- `**tvalues**`: array
    - The t-statistic for the null hypothesis that the corresponding coefficient is zero.
- `**pvalues**`: array
    - The p-values associated with the t-statistics.
- `**cov_params()**`: array
    - Estimated covariance matrix of the parameters.

### **Outputs (from completed model):**

- `**summary()**`: summary instance
    - Provides a summary of the GLM including coefficients, standard errors, t-values, p-values, and other statistics.
- `**params**`: array
    - Estimated coefficients for the GLM.
- `**fittedvalues**`: array
    - The predicted values of the dependent variable based on the fitted GLM.
- `**resid_response**`: array
    - The residuals of the GLM.
- `**predict()**`: method
    - Predict the response variable for new data.

### **When to use** `**statsmodels.api.GLM**`**:**

- **Non-normal response variable:** When the response variable is not normally distributed, such as in count data or binary data.
- **Flexible error distributions:** GLMs allow for the specification of different error distributions (e.g., Gaussian, binomial, Poisson) to accommodate the characteristics of the response variable.
- **Link functions:** GLMs can model the relationship between the mean of the response variable and the linear combination of the independent variables using different link functions, providing flexibility in modeling non-linear relationships.
- **Inference:** GLMs provide inferential statistics such as coefficients, standard errors, t-values, and p-values, which can be used for hypothesis testing and interpretation.

1. **Preparation and Setup**
    - **Parameter Definition:** Establish the number of splits or folds (K) for the validation process, decide on the repetitions for techniques like Repeated K-Fold or K-Split if needed, and outline the hyperparameter grid (C) for each candidate model. For models lacking hyperparameters, set C to an empty configuration.
2. **Data Partitioning**
    - **Validation Holdout (Optional):** Optionally reserve a subset of the data as a standalone validation set (V) to provide an unbiased final evaluation.
3. **Model Evaluation**
    - **K Splits/Folds Validation (Outer Loop):** Organize the data into K distinct splits or folds, using a consistent strategy to ensure each segment of data is used for validation once. This could involve stratified sampling to preserve class distributions in cases of imbalanced datasets.
        - **Hyperparameter Optimization (Inner Loop):** Within each split or fold, perform hyperparameter tuning for models with configurable parameters. This can include nested validation within the training portion of each split or fold to determine the optimal settings.
4. **Performance Aggregation**
    - **Score Compilation:** Collect and average the performance scores across all K splits or folds to derive a comprehensive performance metric for each model configuration.
5. **Model Selection**
    - **Optimal Model Identification:** Evaluate the aggregated performance of each model to select the one that demonstrates the best balance of accuracy, precision, recall, F1 score, or other relevant metrics, considering the specific objectives and constraints of the study.
6. **Final Model Training and Validation**
    - **Comprehensive Training:** Use the entire dataset (excluding any validation holdout) to train the selected model with the identified optimal hyperparameters.
    - **External Validation (Optional):** If a separate validation set was reserved or an external dataset is available, assess the finalized model against this data to gauge its performance and generalizability.
7. **Documentation and Transparency**
    - **In-depth Reporting:** Thoroughly document the selection process, including the rationale behind the choice of metrics, models, hyperparameters, and the comparative performance across different model configurations, to ensure clarity and reproducibility.

### **Additional Considerations**

- **Handling Class Imbalances:** Implement appropriate strategies, such as weighted training or stratified sampling, throughout the validation process to address imbalanced data.
- **Computational Efficiency:** Be cognizant of the resource demands, especially with extensive model grids and validation strategies. Employ efficient search techniques and parallel processing where feasible.
- **Adaptation to Domain Requirements:** Customize the model selection framework to align with the domain-specific needs and the nature of the data, ensuring the chosen approach is both relevant and practical.