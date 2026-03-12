# .github/tests/test_assignment5.py
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import nbformat
import re
import joblib # Using joblib to save/load models if needed, or rely on notebook execution state
import subprocess # To run generate_data.py
from nbclient import NotebookClient # To execute notebooks
from nbclient.exceptions import CellExecutionError

# --- Helper Function to Load Notebook Functions ---
# (Keep this helper, it's useful)
def load_notebook_functions(notebook_path):
    """Extracts Python functions defined in a Jupyter notebook."""
    if not Path(notebook_path).is_file():
        print(f"Warning: Notebook '{notebook_path}' not found.")
        return {} # Return empty dict if notebook doesn't exist

    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            nb = nbformat.read(f, as_version=4)
        except Exception as e:
            print(f"Error reading notebook {notebook_path}: {e}")
            return {}

    functions = {}
    full_code = ""
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Append all code cells to execute them in order
            full_code += cell.source + "\n\n"

    # Execute the combined code in a single namespace
    namespace = {}
    try:
        exec(full_code, namespace)
    except Exception as e:
        print(f"Error executing code from {notebook_path}: {e}")
        # Continue to see if *some* functions were defined before the error
        pass # Allow partial loading if possible

    # Extract functions from the namespace
    for name, obj in namespace.items():
        if callable(obj) and isinstance(obj, type(lambda: 0)): # Check if it's a function
             # Basic check to avoid importing classes if not intended
            if not name.startswith('_') and name[0].islower(): # Simple heuristic for functions
                functions[name] = obj

    # Debug: Print loaded functions
    # print(f"Functions loaded from {notebook_path}: {list(functions.keys())}")
    return functions

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def synthetic_data():
    """Loads the generated synthetic data."""
    data_path = Path("data/synthetic_health_data.csv")
    if not data_path.is_file():
        pytest.skip("Synthetic data file not found. Run generate_data.py first.")
    try:
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        return df
    except Exception as e:
        pytest.fail(f"Failed to load synthetic data: {e}")

@pytest.fixture(scope="session", autouse=True)
def run_notebooks_once(request):
    """Fixture to run data generation and execute notebooks once per session."""
    print("\n--- Running Pre-Test Setup ---")
    data_path = Path("data/synthetic_health_data.csv")
    generate_script = "generate_data.py"
    notebooks_to_run = [
        'part1_introduction.ipynb',
        'part2_feature_engineering.ipynb',
        'part3_data_preparation.ipynb'
    ]

    # 1. Generate data if it doesn't exist
    if not data_path.is_file():
        print(f"Data file '{data_path}' not found. Running {generate_script}...")
        try:
            # Ensure necessary packages for the script are available in the test env
            # Note: The workflow installs pandas/numpy, but local runs might need them
            result = subprocess.run(["python", generate_script], capture_output=True, text=True, check=True)
            print(f"Data generation script output:\n{result.stdout}")
            if not data_path.is_file():
                 pytest.fail(f"Script {generate_script} ran but did not create {data_path}")
        except FileNotFoundError:
             pytest.fail(f"Python executable not found. Cannot run {generate_script}.")
        except subprocess.CalledProcessError as e:
             pytest.fail(f"Failed to run {generate_script}: {e}\nOutput:\n{e.stderr}")
        except Exception as e:
             pytest.fail(f"An unexpected error occurred while trying to run {generate_script}: {e}")
    else:
        print(f"Data file '{data_path}' found.")

    # 2. Execute notebooks
    print("Executing notebooks...")
    execution_errors = []
    for nb_path_str in notebooks_to_run:
        nb_path = Path(nb_path_str)
        if not nb_path.is_file():
            print(f"Warning: Notebook '{nb_path}' not found, skipping execution.")
            continue

        print(f"  Executing {nb_path}...")
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Configure the client (use the kernel installed by the workflow/locally)
            # Ensure the kernel name matches what's available/installed
            # Defaulting to 'python3' which is common
            client = NotebookClient(nb, timeout=600, kernel_name='python3')

            # Execute the notebook in the current directory
            client.execute()

            # Optional: Save the executed notebook (useful for debugging)
            # with open(f"{nb_path.stem}_executed.ipynb", 'w', encoding='utf-8') as f:
            #     nbformat.write(nb, f)
            print(f"  Finished executing {nb_path}.")

        except CellExecutionError as e:
            # Capture cell execution errors specifically
            print(f"ERROR during execution of {nb_path}:\n{e}")
            # Optionally, include traceback or specific cell info if needed
            execution_errors.append(f"Error in {nb_path}: {e}")
        except Exception as e:
            # Catch other potential errors (like kernel not found)
            print(f"Failed to execute notebook {nb_path}: {e}")
            execution_errors.append(f"Failed {nb_path}: {e}")

    print("--- Finished Pre-Test Setup ---")

    # If any notebook failed, skip all tests
    if execution_errors:
        pytest.fail(f"Notebook execution failed, skipping tests:\n" + "\n".join(execution_errors), pytrace=False)


# --- Test Classes ---

class TestPart1:
    notebook_path = 'part1_introduction.ipynb'
    functions = load_notebook_functions(notebook_path)

    @pytest.fixture(autouse=True)
    def check_functions_loaded(self):
        if not self.functions:
             pytest.skip(f"Could not load functions from {self.notebook_path}. Skipping tests.")

    def test_p1_load_data(self, synthetic_data):
        """Test Part 1 data loading."""
        load_data = self.functions.get('load_data')
        if load_data is None:
            pytest.skip("load_data function not found in part1 notebook")

        # Test loading the actual generated data
        df = load_data("data/synthetic_health_data.csv")
        assert isinstance(df, pd.DataFrame), "load_data should return a DataFrame"
        assert not df.empty, "Loaded DataFrame should not be empty"
        expected_cols = ['patient_id', 'timestamp', 'age', 'systolic_bp', 'diastolic_bp',
                         'glucose_level', 'bmi', 'smoker_status', 'heart_rate', 'disease_outcome']
        assert all(col in df.columns for col in expected_cols), "Missing expected columns"
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "Timestamp column should be datetime"

    def test_p1_prepare_data(self, synthetic_data):
        """Test Part 1 data preparation."""
        prepare_data_part1 = self.functions.get('prepare_data_part1')
        if prepare_data_part1 is None:
            pytest.skip("prepare_data_part1 function not found in part1 notebook")

        X_train, X_test, y_train, y_test = prepare_data_part1(synthetic_data.copy())

        assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
        assert isinstance(y_train, pd.Series), "y_train should be a Series"
        assert isinstance(y_test, pd.Series), "y_test should be a Series"

        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train length mismatch"
        assert X_test.shape[0] == y_test.shape[0], "X_test and y_test length mismatch"
        assert X_train.shape[0] > X_test.shape[0], "Train set should be larger than test set"

        expected_features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
        assert all(f in X_train.columns for f in expected_features), "X_train missing expected features"
        assert all(f in X_test.columns for f in expected_features), "X_test missing expected features"

        assert not X_train.isnull().any().any(), "X_train should not contain NaNs after imputation"
        assert not X_test.isnull().any().any(), "X_test should not contain NaNs after imputation"

    def test_p1_train_logistic_regression(self, synthetic_data):
        """Test Part 1 logistic regression training."""
        prepare_data_part1 = self.functions.get('prepare_data_part1')
        train_logistic_regression = self.functions.get('train_logistic_regression')
        if prepare_data_part1 is None or train_logistic_regression is None:
            pytest.skip("Required functions (prepare/train) not found in part1 notebook")

        X_train, _, y_train, _ = prepare_data_part1(synthetic_data.copy())
        model = train_logistic_regression(X_train, y_train)

        from sklearn.linear_model import LogisticRegression
        assert isinstance(model, LogisticRegression), "Should return a LogisticRegression model"
        assert hasattr(model, 'predict'), "Model should have a predict method"
        assert hasattr(model, 'predict_proba'), "Model should have a predict_proba method"

    # Test that the results file is created and contains valid metrics
    def test_p1_results_file(self):
        """Test Part 1 results file generation and content."""
        # This test relies on the run_notebooks_once fixture having executed the notebook
        results_file = Path("results/results_part1.txt")
        assert results_file.is_file(), f"{results_file} not found. Ensure the notebook saves results."

        results = {}
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        results[key.strip()] = float(value.strip())
        except Exception as e:
            pytest.fail(f"Error reading or parsing {results_file}: {e}")

        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        assert all(key in results for key in expected_metrics), \
            f"Missing expected metrics in {results_file}. Found: {list(results.keys())}"

        for metric, value in results.items():
             if metric in expected_metrics: # Only check expected metrics
                 assert 0 <= value <= 1, f"Metric '{metric}' value {value} is out of range [0, 1] in {results_file}"


class TestPart2:
    notebook_path = 'part2_feature_engineering.ipynb'
    functions = load_notebook_functions(notebook_path)

    @pytest.fixture(autouse=True)
    def check_functions_loaded(self):
        if not self.functions:
             pytest.skip(f"Could not load functions from {self.notebook_path}. Skipping tests.")

    def test_p2_extract_rolling_features(self, synthetic_data):
        """Test Part 2 rolling feature extraction."""
        extract_rolling_features = self.functions.get('extract_rolling_features')
        if extract_rolling_features is None:
            pytest.skip("extract_rolling_features function not found in part2 notebook")

        window_size = 300 # Example window size in seconds
        df_featured = extract_rolling_features(synthetic_data.copy(), window_size)

        assert isinstance(df_featured, pd.DataFrame), "Should return a DataFrame"
        assert 'hr_rolling_mean' in df_featured.columns, "Missing hr_rolling_mean column"
        assert 'hr_rolling_std' in df_featured.columns, "Missing hr_rolling_std column"
        assert not df_featured['hr_rolling_mean'].isnull().any(), "hr_rolling_mean should not have NaNs"
        # Initial std might be NaN if min_periods > 1, but should be filled
        assert not df_featured['hr_rolling_std'].isnull().any(), "hr_rolling_std should not have NaNs (should be filled)"

    def test_p2_prepare_data(self, synthetic_data):
        """Test Part 2 data preparation."""
        extract_rolling_features = self.functions.get('extract_rolling_features')
        prepare_data_part2 = self.functions.get('prepare_data_part2')
        if extract_rolling_features is None or prepare_data_part2 is None:
            pytest.skip("Required functions (extract/prepare) not found in part2 notebook")

        df_featured = extract_rolling_features(synthetic_data.copy(), 300)
        X_train, X_test, y_train, y_test = prepare_data_part2(df_featured)

        assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"

        expected_features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi',
                             'hr_rolling_mean', 'hr_rolling_std']
        assert all(f in X_train.columns for f in expected_features), "X_train missing expected features"
        assert not X_train.isnull().any().any(), "X_train_pt2 should not contain NaNs"
        assert not X_test.isnull().any().any(), "X_test_pt2 should not contain NaNs"

    def test_p2_train_random_forest(self, synthetic_data):
        """Test Part 2 Random Forest training."""
        extract_rolling_features = self.functions.get('extract_rolling_features')
        prepare_data_part2 = self.functions.get('prepare_data_part2')
        train_random_forest = self.functions.get('train_random_forest')
        if extract_rolling_features is None or prepare_data_part2 is None or train_random_forest is None:
            pytest.skip("Required functions (extract/prepare/train_rf) not found in part2 notebook")

        df_featured = extract_rolling_features(synthetic_data.copy(), 300)
        X_train, _, y_train, _ = prepare_data_part2(df_featured)
        model = train_random_forest(X_train, y_train)

        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier), "Should return a RandomForestClassifier model"
        assert hasattr(model, 'predict'), "Model should have a predict method"

    def test_p2_train_xgboost(self, synthetic_data):
        """Test Part 2 XGBoost training."""
        extract_rolling_features = self.functions.get('extract_rolling_features')
        prepare_data_part2 = self.functions.get('prepare_data_part2')
        train_xgboost = self.functions.get('train_xgboost')
        if extract_rolling_features is None or prepare_data_part2 is None or train_xgboost is None:
            pytest.skip("Required functions (extract/prepare/train_xgb) not found in part2 notebook")

        df_featured = extract_rolling_features(synthetic_data.copy(), 300)
        X_train, _, y_train, _ = prepare_data_part2(df_featured)
        model = train_xgboost(X_train, y_train)

        import xgboost as xgb
        assert isinstance(model, xgb.XGBClassifier), "Should return an XGBClassifier model"
        assert hasattr(model, 'predict'), "Model should have a predict method"

    # Test that the results file is created and contains valid AUCs
    def test_p2_results_file(self):
        """Test Part 2 results file generation and content."""
        # This test relies on the run_notebooks_once fixture having executed the notebook
        results_file = Path("results/results_part2.txt")
        assert results_file.is_file(), f"{results_file} not found. Ensure the notebook saves results."

        results = {}
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        results[key.strip()] = float(value.strip())
        except Exception as e:
            pytest.fail(f"Error reading or parsing {results_file}: {e}")

        expected_metrics = ['rf_auc', 'xgb_auc']
        assert all(key in results for key in expected_metrics), \
            f"Missing expected metrics in {results_file}. Found: {list(results.keys())}"

        for metric, value in results.items():
            if metric in expected_metrics:
                assert 0 <= value <= 1, f"Metric '{metric}' value {value} is out of range [0, 1] in {results_file}"


class TestPart3:
    notebook_path = 'part3_data_preparation.ipynb'
    functions = load_notebook_functions(notebook_path)

    @pytest.fixture(autouse=True)
    def check_functions_loaded(self):
        if not self.functions:
             pytest.skip(f"Could not load functions from {self.notebook_path}. Skipping tests.")

    # Test encode_categorical_features just for basic transformation
    def test_p3_encode_categorical_features(self, synthetic_data):
         """Test Part 3 categorical encoding function (basic check)."""
         encode_categorical_features = self.functions.get('encode_categorical_features')
         if encode_categorical_features is None:
             pytest.skip("encode_categorical_features function not found in part3 notebook")

         df_encoded = encode_categorical_features(synthetic_data.copy())
         assert isinstance(df_encoded, pd.DataFrame), "Should return a DataFrame"
         assert 'smoker_status' not in df_encoded.columns, "Original smoker_status column should be removed"
         # Check if new columns exist (names might vary slightly based on implementation)
         assert any(col.startswith('smoker_status_') for col in df_encoded.columns), "Expected one-hot encoded columns for smoker_status"
         # Check that the number of rows is the same
         assert len(df_encoded) == len(synthetic_data), "Number of rows should not change"


    def test_p3_prepare_data(self, synthetic_data):
        """Test Part 3 data preparation including correct encoding."""
        prepare_data_part3 = self.functions.get('prepare_data_part3')
        if prepare_data_part3 is None:
            pytest.skip("prepare_data_part3 function not found in part3 notebook")

        X_train, X_test, y_train, y_test, encoder = prepare_data_part3(synthetic_data.copy())

        assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
        assert 'smoker_status' not in X_train.columns, "Original smoker_status should be removed from X_train"
        assert 'smoker_status' not in X_test.columns, "Original smoker_status should be removed from X_test"
        assert any(col.startswith('smoker_status_') for col in X_train.columns), "X_train missing encoded smoker_status columns"
        assert any(col.startswith('smoker_status_') for col in X_test.columns), "X_test missing encoded smoker_status columns"
        assert not X_train.isnull().any().any(), "X_train_pt3 should not contain NaNs"
        assert not X_test.isnull().any().any(), "X_test_pt3 should not contain NaNs"

        from sklearn.preprocessing import OneHotEncoder
        assert isinstance(encoder, OneHotEncoder), "Should return the fitted OneHotEncoder"


    def test_p3_apply_smote(self, synthetic_data):
        """Test Part 3 SMOTE application."""
        prepare_data_part3 = self.functions.get('prepare_data_part3')
        apply_smote = self.functions.get('apply_smote')
        if prepare_data_part3 is None or apply_smote is None:
            pytest.skip("Required functions (prepare/smote) not found in part3 notebook")

        X_train, _, y_train, _ , _= prepare_data_part3(synthetic_data.copy())
        original_minority_count = y_train.sum() # Assuming 1 is minority

        X_res, y_res = apply_smote(X_train, y_train)

        assert isinstance(X_res, pd.DataFrame) or isinstance(X_res, np.ndarray), "X_res should be DataFrame or ndarray"
        assert isinstance(y_res, pd.Series) or isinstance(y_res, np.ndarray), "y_res should be Series or ndarray"
        assert len(X_res) == len(y_res), "Resampled X and y length mismatch"
        assert len(X_res) > len(X_train), "SMOTE should increase the number of samples"

        # Check if minority class count increased significantly
        resampled_minority_count = pd.Series(y_res).sum() # Convert to series if numpy array
        assert resampled_minority_count > original_minority_count, "Minority class count should increase after SMOTE"
        # Check if classes are now (approximately) balanced
        assert np.isclose(pd.Series(y_res).value_counts(normalize=True)[0], 0.5, atol=0.01), "Classes should be roughly balanced after SMOTE"

    # Test that the results file is created and contains valid metrics
    def test_p3_results_file(self):
        """Test Part 3 results file generation and content."""
        # This test relies on the run_notebooks_once fixture having executed the notebook
        results_file = Path("results/results_part3.txt")
        assert results_file.is_file(), f"{results_file} not found. Ensure the notebook saves results."

        results = {}
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value_str = line.split(':', 1)
                        key = key.strip()
                        value_str = value_str.strip()
                        # Handle potential NaN for AUC
                        if value_str.lower() == 'nan':
                            results[key] = np.nan
                        else:
                            results[key] = float(value_str)
        except Exception as e:
            pytest.fail(f"Error reading or parsing {results_file}: {e}")

        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        assert all(key in results for key in expected_metrics), \
            f"Missing expected metrics in {results_file}. Found: {list(results.keys())}"

        for metric, value in results.items():
             if metric in expected_metrics and not np.isnan(value): # Only check non-NaN expected metrics
                 assert 0 <= value <= 1, f"Metric '{metric}' value {value} is out of range [0, 1] in {results_file}"


if __name__ == "__main__":
    # The run_notebooks_once fixture handles data generation now
    # You might still want a local run trigger, but the fixture covers the core need
    print("Running tests locally. Data generation and notebook execution handled by pytest fixture.")
    pytest.main(["-v", __file__])