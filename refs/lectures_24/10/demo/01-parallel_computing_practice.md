# Demo 1: Parallel Computing Practice

**Objective:** Experience the power of parallel computing by analyzing patient data, observe real speedup, and create your first SLURM script. We'll use `concurrent.futures` for notebook-friendly parallelism and discuss `multiprocessing.Pool` for standalone scripts.

## Step 1: Local Parallel Analysis

### 1.1 Setup: Imports and Configuration

First, let's import necessary libraries and set up our `DEBUG_MODE` flag. This flag allows us to run a "baby-sized" version of the demo for quick testing or a "full-sized" version to observe more significant performance differences.

```python
import pandas as pd
import numpy as np
import os
import time
import sys # For checking if running in a notebook
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor # For parallel execution

# DEBUG MODE: Set to True for faster testing, False for full analysis
DEBUG_MODE = True 

# Determine if running in a Jupyter-like environment for nbconvert robustness
# 'get_ipython' is a reliable check for IPython/Jupyter environments.
IS_NOTEBOOK_ENVIRONMENT = 'get_ipython' in globals() or hasattr(sys, 'ps1')

print(f"DEBUG_MODE: {DEBUG_MODE}")
print(f"IS_NOTEBOOK_ENVIRONMENT: {IS_NOTEBOOK_ENVIRONMENT}")
```

### 1.2 Create Sample Patient Data Files

Next, we'll generate some sample patient cohort data. Each cohort will be a CSV file. The size and number of these files will depend on our `DEBUG_MODE`.

```python
# Configure based on mode
if DEBUG_MODE:
    n_files = 4          # Only 4 files for quick testing
    min_patients = 100   # Smaller cohorts (100-500 patients)
    max_patients = 500
    print("🐣 DEBUG MODE: Creating baby-sized dataset for quick testing...")
else:
    n_files = 20         # Full 20 files for more noticeable speedup
    min_patients = 1000  # Full-size cohorts (1K-5K patients)  
    max_patients = 5000
    print("🚀 FULL MODE: Creating complete dataset...")

# Define the directory for patient data
# Ensures it's created relative to the current working directory of the notebook/script
data_dir = 'patient_cohorts_demo_data' # Use a distinct name
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created directory: {data_dir}")
else:
    print(f"Directory already exists: {data_dir}")


# Generate sample patient cohort files
current_patient_files = [] # List to store paths of created files
for i in range(n_files):
    n_patients = np.random.randint(min_patients, max_patients)
    file_path = os.path.join(data_dir, f'cohort_{i:02d}.csv')
    
    patients = pd.DataFrame({
        'patient_id': range(n_patients),
        'age': np.random.normal(65, 15, n_patients),
        'risk_score': np.random.beta(2, 5, n_patients),
        'hospital_days': np.random.poisson(3, n_patients),
        'comorbidities': np.random.poisson(2, n_patients)
    })
    
    patients.to_csv(file_path, index=False)
    current_patient_files.append(file_path)
    
print(f"✅ Created {len(current_patient_files)} patient cohort files in '{data_dir}'.")
if current_patient_files:
    print(f"Example file path: {current_patient_files[0]}")
```

### 1.3 Define the Core Analysis Function

This is the heart of our computation. The `analyze_patient_cohort` function will:
1. Read a patient cohort CSV file into a pandas DataFrame.
2. Simulate a somewhat complex, time-consuming analysis using `time.sleep()`. In a real scenario, this could be model training, complex statistical calculations, or data transformations.
3. Return a dictionary of summary statistics.

The `DEBUG_MODE` flag will control the duration of `time.sleep()` to allow for faster runs during development and testing.

```python
def analyze_patient_cohort(patient_file_path):
    """
    Analyzes a single patient cohort CSV file.
    Simulates a time-consuming task.
    """
    # print(f"Processing {patient_file_path} in process/thread {os.getpid()}/{threading.get_ident()}") # Optional debug
    try:
        df = pd.read_csv(patient_file_path)
        
        # Simulate complex analysis - duration depends on DEBUG_MODE
        sleep_duration = 0.02 if DEBUG_MODE else 0.2 
        time.sleep(sleep_duration)
        
        return {
            'file': patient_file_path,
            'n_patients': len(df),
            'avg_age': df['age'].mean(),
            'high_risk_count': (df['risk_score'] > 0.8).sum(),
            'status': 'success'
        }
    except Exception as e:
        # print(f"Error processing {patient_file_path}: {e}") # Optional debug
        return {
            'file': patient_file_path,
            'status': 'error',
            'error_message': str(e)
        }

# Quick test of the function with the first generated file (if available)
if 'current_patient_files' in globals() and current_patient_files:
    print(f"🧪 Testing 'analyze_patient_cohort' with one file: {current_patient_files[0]}")
    try:
        # Need to ensure DEBUG_MODE is accessible if this cell is run alone
        if 'DEBUG_MODE' not in globals(): DEBUG_MODE = True # Fallback for isolated cell run
        
        single_result = analyze_patient_cohort(current_patient_files[0])
        if single_result['status'] == 'success':
            print(f"👍 Test successful. Result for {single_result['file']}: Patients: {single_result['n_patients']}")
        else:
            print(f"❌ Test failed for {single_result['file']}: {single_result.get('error_message', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
else:
    print("⚠️ 'current_patient_files' list not found or empty. Please ensure the data generation cell was run successfully.")
```

### 1.4 Sequential Processing: Establishing a Baseline

Before we explore parallel execution, let's run our analysis sequentially. This means processing each file one after the other, using a single line of execution. This will give us a baseline time to compare against.

```python
# Ensure 'current_patient_files' and 'analyze_patient_cohort' are available
if 'current_patient_files' not in globals() or not current_patient_files:
    print("⚠️ 'current_patient_files' not defined or empty. Cannot run sequential analysis.")
    sequential_time = -1 # Indicate an error
    sequential_results = []
    execution_times = {} # Initialize if not present
else:
    print(f"⏳ Starting sequential analysis of {len(current_patient_files)} files...")
    
    start_time_seq = time.time()
    sequential_results = [analyze_patient_cohort(f) for f in current_patient_files]
    sequential_time = time.time() - start_time_seq
    
    print(f"✅ Sequential analysis completed in: {sequential_time:.3f} seconds.")
    
    # Store this time for later comparison
    if 'execution_times' not in globals():
        execution_times = {} # Initialize if it wasn't created due to prior error
    execution_times['Sequential (1 worker)'] = sequential_time
```

### 1.5 Parallel Processing: Comparing Different Numbers of Workers

Now, let's harness the power of multiple workers! We'll use `ThreadPoolExecutor` from the `concurrent.futures` module, which is generally robust for use in Jupyter Notebooks (especially when `nbconvert --execute` is used). For CPU-bound tasks like our `time.sleep()` simulation, `ProcessPoolExecutor` would typically provide better speedups in standalone Python scripts by bypassing the Global Interpreter Lock (GIL), but `ThreadPoolExecutor` is safer for notebook execution and still demonstrates concurrency.

We will test with the following numbers of `max_workers`:
- **4 workers**
- **8 workers**
- **"Full" workers**: This will be `os.cpu_count()` (the number of logical CPUs your system reports), but capped at a maximum of 16 for this demonstration to keep results comparable and avoid excessive resource use on machines with many cores. If `DEBUG_MODE` is active, this "full" count will be further capped at 4.

We'll collect the execution time for each configuration and then display a summary.

```python
# Ensure required variables are available
if 'current_patient_files' not in globals() or \
   'analyze_patient_cohort' not in globals() or \
   'DEBUG_MODE' not in globals() or \
   'execution_times' not in globals():
    print("⚠️ One or more required variables not found. Please run previous cells before this one.")
else:
    # Determine the number of workers for "full" parallelism for the demo
    cpu_cores = os.cpu_count() or 1 # Default to 1 if os.cpu_count() is None or 0
    
    # Max workers for the "full" test: min(cpu_cores, 16)
    # If DEBUG_MODE, cap this further to min(current_cap, 4)
    max_workers_full_demo = min(cpu_cores, 16) 
    if DEBUG_MODE:
        max_workers_full_demo = min(max_workers_full_demo, 4)

    # Define the specific worker configurations to test
    # We want to test 4, 8, and our determined "full" (e.g., 16 or less)
    worker_test_configurations = [4, 8, max_workers_full_demo]
    
    # Filter: only include counts <= max_workers_full_demo, ensure positive, unique, and sort
    worker_test_configurations = sorted(list(set(w for w in worker_test_configurations if 0 < w <= max_workers_full_demo)))
    
    # If the list became empty (e.g., max_workers_full_demo < 4), add at least one valid config
    if not worker_test_configurations and max_workers_full_demo > 0:
        worker_test_configurations.append(max_workers_full_demo)
    elif not worker_test_configurations: # If max_workers_full_demo was 0 or less
        worker_test_configurations.append(1) # Default to 1 worker

    print(f"⚙️ System CPU cores reported: {cpu_cores}.")
    print(f"🛠️ Max workers for 'full' demo configuration: {max_workers_full_demo}.")
    print(f"🧪 Will test parallel execution with worker counts: {worker_test_configurations}")

    # Choose executor based on environment (ThreadPool for notebooks, can switch to ProcessPool for scripts)
    # For nbconvert --execute, ThreadPoolExecutor is generally more reliable.
    executor_to_use = ThreadPoolExecutor
    executor_name_for_log = "ThreadPoolExecutor"
    if not IS_NOTEBOOK_ENVIRONMENT:
        # In a script, ProcessPoolExecutor might be preferred for CPU-bound tasks
        # executor_to_use = ProcessPoolExecutor 
        # executor_name_for_log = "ProcessPoolExecutor"
        print(f"   (Running as script, using {executor_name_for_log}. Consider ProcessPoolExecutor for CPU-bound tasks if not already selected.)")
    else:
        print(f"   (Running in notebook, using {executor_name_for_log} for robustness with nbconvert.)")


    for n_workers in worker_test_configurations:
        print(f"\n⚡ Running parallel analysis with {n_workers} workers using {executor_name_for_log}...")
        start_time_parallel_run = time.time()
        
        try:
            with executor_to_use(max_workers=n_workers) as executor:
                parallel_results = list(executor.map(analyze_patient_cohort, current_patient_files))
            current_run_time = time.time() - start_time_parallel_run
            execution_times[f'Parallel ({n_workers} workers)'] = current_run_time
            print(f"✅ Parallel ({n_workers} workers) completed in: {current_run_time:.3f}s")
        except Exception as e:
            print(f"❌ Error during parallel execution with {n_workers} workers: {e}")
            execution_times[f'Parallel ({n_workers} workers)'] = -1 # Indicate error

    # --- Summary of Execution Times ---
    print("\n\n📊📊📊 Execution Time Summary 📊📊📊")
    print(f"{'Configuration':<45} | {'Time (s)':<10} | {'Speedup vs Sequential':<20}")
    print(f"{'-'*80}")
    
    base_sequential_time = execution_times.get('Sequential (1 worker)', 0)
    
    if base_sequential_time > 0:
        print(f"{'Sequential (1 worker)':<45} | {base_sequential_time:<10.3f} | {'1.00x':<20}")
        
        # Sort parallel results by number of workers for clearer presentation
        parallel_keys = sorted([k for k in execution_times if k.startswith('Parallel')], 
                               key=lambda x: int(x.split('(')[1].split(' ')[0]))

        for config_key in parallel_keys:
            time_taken = execution_times[config_key]
            if time_taken > 0:
                speedup = base_sequential_time / time_taken
                print(f"{config_key:<45} | {time_taken:<10.3f} | {speedup:<20.2f}x")
            else:
                print(f"{config_key:<45} | {'N/A (error)':<10} | {'N/A':<20}")
    else:
        print("Sequential time not available or zero, speedup calculation skipped. Please ensure the sequential cell ran correctly.")
    print(f"{'-'*80}")
```

---

### Alternative: `multiprocessing.Pool` (Primarily for Standalone `.py` Scripts)

The `multiprocessing.Pool` is a powerful tool for parallelism, especially for CPU-bound tasks in standalone Python scripts, as it can bypass the Global Interpreter Lock (GIL) by using separate processes. However, it often presents challenges in Jupyter Notebooks, particularly with function pickling and the `if __name__ == '__main__':` guard requirement on some operating systems (like Windows and macOS when using 'spawn' or 'forkserver' start methods).

The cell below shows how you might use `multiprocessing.Pool`. **It is wrapped in `if False:` to prevent execution by `nbconvert` and is intended as an example for students to adapt into a `.py` file.**

**To run this example:**
1.  Copy the `analyze_patient_cohort` function definition and the `DEBUG_MODE` variable from above into a new Python file (e.g., `run_mp_pool_demo.py`).
2.  Copy the code block below (inside the `if __name__ == '__main__':`) into that same `.py` file.
3.  Run the script from your terminal: `python run_mp_pool_demo.py`

```python
# THIS CELL IS INTENDED FOR .py SCRIPT EXECUTION AND EXPLANATION.
# It will NOT run correctly if executed directly in a Jupyter Notebook cell by nbconvert.
# The 'if False:' block ensures it's skipped during notebook execution.

if False: 
    import multiprocessing as mp
    # Assume pandas, time, os, np are imported in the .py file
    # Assume DEBUG_MODE and analyze_patient_cohort are defined at the top-level of the .py file

    # --- Content for your .py file ---
    # DEBUG_MODE = True # Or False
    #
    # def analyze_patient_cohort(patient_file_path):
    #     # ... (full function definition as above) ...
    #     pass
    #
    # current_patient_files = [...] # Define this list, e.g., by recreating it
    # --- End of content for .py file ---

    if __name__ == '__main__':  # CRITICAL for multiprocessing.Pool on some OSes!
        # This guard ensures that the pool is only created when the script is run directly,
        # not when it's imported by child processes.
        
        # Re-create necessary variables if they are not globally defined in the script
        # For example, if DEBUG_MODE and current_patient_files are from notebook cells:
        script_debug_mode = True # Set as needed for the script
        script_n_files = 4 if script_debug_mode else 20
        
        # Ensure data_dir is defined or paths are correct for the script's context
        script_data_dir = 'patient_cohorts_demo_data' 
        if not os.path.exists(script_data_dir): # Create data if script is run standalone
             os.makedirs(script_data_dir)
             # Simplified data generation for script example
             for i in range(script_n_files):
                 pd.DataFrame({'data': [np.random.rand()]}).to_csv(os.path.join(script_data_dir, f'cohort_{i:02d}.csv'))

        script_patient_files = [os.path.join(script_data_dir, f'cohort_{i:02d}.csv') for i in range(script_n_files)]


        print(f"📊 (multiprocessing.Pool Example) Analyzing {len(script_patient_files)} cohort files...")
        
        # Ensure analyze_patient_cohort is picklable (defined at top-level of the module)
        # and DEBUG_MODE is accessible (e.g., global in the script or passed via initializer)

        start_time_mp_pool = time.time()
        try:
            # Using a fixed number of processes for this example
            with mp.Pool(processes=4) as pool: 
                results_mp_pool = pool.map(analyze_patient_cohort, script_patient_files)
            time_mp_pool = time.time() - start_time_mp_pool
            print(f"✅ Parallel (multiprocessing.Pool with 4 processes) completed in: {time_mp_pool:.3f}s")
        except Exception as e_mp:
            print(f"❌ Error with multiprocessing.Pool: {e_mp}")
            print("   This often happens in notebooks. Ensure this code is in a .py file,")
            print("   'analyze_patient_cohort' is defined at the top level, and the")
            print("   `if __name__ == '__main__':` guard is used.")
    # --- End of example for .py file ---
```

## Step 2: SLURM Script Creation (5 minutes)

Now that we've seen how to parallelize code locally, let's prepare a script for running on a High-Performance Computing (HPC) cluster using SLURM. SLURM is a workload manager that schedules jobs on the cluster.

Create a new file named `my_health_analysis.sh` in the same directory as this notebook (or your `.py` script) with the following content. This script tells SLURM how many resources (CPUs, memory, time) your job needs and what commands to run.

```bash
#!/bin/bash
#SBATCH --job-name=my_first_health_analysis  # Name of your job
#SBATCH --cpus-per-task=4                  # Request 4 CPUs
#SBATCH --mem=16G                          # Request 16GB of memory
#SBATCH --time=01:00:00                    # Request 1 hour of wall-clock time
#SBATCH --output=logs/health_analysis_%j.out # Where to save standard output (%j is job ID)
#SBATCH --error=logs/health_analysis_%j.err  # Where to save standard error

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================================"
echo "Job Started: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Number of CPUs requested: $SLURM_CPUS_PER_TASK"
echo "Memory requested: $SLURM_MEM_PER_TASK MB per CPU (if specified, else total $SLURM_MEM_PER_NODE MB)"
echo "========================================================"

# Load necessary modules (example for Wynton HPC, adjust for your cluster)
# module load python/3.9  # Or your preferred Python version
# module load anaconda    # Or if you use Anaconda environments

# Activate your Python virtual environment (if you use one)
# source /path/to/your/.venv/bin/activate 

# Navigate to the directory containing your script (if necessary)
# cd /path/to/your/script_directory 

# Command to run your Python script
# This script should be the one using multiprocessing.Pool or concurrent.futures
# For this demo, assume you have a 'parallel_patient_analysis.py'
# You would create this .py file from the notebook cells above.
echo "Running Python script for parallel analysis..."
python parallel_patient_analysis.py # Replace with your actual script name

echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================"
```

**Note on `parallel_patient_analysis.py`**: You would need to create this Python script. It would contain:
1.  The `DEBUG_MODE` flag (likely set to `False` for cluster runs).
2.  The `analyze_patient_cohort` function.
3.  The main execution block (e.g., using `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`) guarded by `if __name__ == '__main__':`.

## Success Criteria

✅ **Speedup Observed:** Parallel version runs noticeably faster than sequential, especially with `DEBUG_MODE = False`.
✅ **Understanding Gained:** Can explain when to use `ThreadPoolExecutor` vs. `ProcessPoolExecutor` vs. `multiprocessing.Pool`.
✅ **SLURM Ready:** Created a basic SLURM script for cluster submission.
✅ **Resource Awareness:** Understand the relationship between `max_workers` and potential performance.

> 💡 **Debug Tip:** Start with `DEBUG_MODE = True` to quickly test your setup and code logic. Then switch to `DEBUG_MODE = False` to observe more realistic performance differences with larger datasets and longer simulated computation times.

## Common Issues & Solutions

- **"No speedup observed" (or slowdowns):**
    -   **Task Granularity:** If `sleep_duration` is too short (like in `DEBUG_MODE`), the overhead of creating and managing threads/processes can outweigh the benefits. Try with `DEBUG_MODE = False`.
    -   **Global Interpreter Lock (GIL) with `ThreadPoolExecutor`:** For purely CPU-bound Python code, `ThreadPoolExecutor` won't provide true parallelism due to the GIL. `ProcessPoolExecutor` or `multiprocessing.Pool` (in a `.py` script) are needed to bypass this. Our `time.sleep()` actually releases the GIL, so `ThreadPoolExecutor` *can* show speedup here, mimicking I/O-bound tasks.
    -   **Too many workers:** Creating too many workers can lead to excessive context switching.
- **"Memory errors":** If processing very large files, reduce `n_files` or the data size within each file, or request more memory in your SLURM script.
- **`AttributeError: Can't get attribute 'analyze_patient_cohort' on <module '__main__'...>` (with `ProcessPoolExecutor` or `mp.Pool` in notebooks):** This is a common pickling issue in Jupyter. Functions to be run by child processes need to be importable or defined in a way that child processes can find them. `ThreadPoolExecutor` is usually safer in notebooks. For `ProcessPoolExecutor` or `mp.Pool`, ensure the function is defined at the top level of a module (a `.py` file) if issues persist.
- **`RuntimeError` about bootstrapping / `if __name__ == '__main__':`:** This is crucial for `multiprocessing.Pool` (and sometimes `ProcessPoolExecutor` depending on the OS and start method) when running scripts. It prevents child processes from re-executing the main script's code that spawns them.
- **`NameError: name 'DEBUG_MODE' is not defined` in `analyze_patient_cohort`:** Ensure `DEBUG_MODE` is defined in a scope accessible to the function when it's called by parallel workers. For `ProcessPoolExecutor` or `mp.Pool`, this usually means it should be a global variable in the module where `analyze_patient_cohort` is defined.

## 💡 Pro Tips

- **`concurrent.futures` vs. `multiprocessing.Pool`**:
    -   `concurrent.futures` (both `ThreadPoolExecutor` and `ProcessPoolExecutor`) offers a more modern, higher-level API.
    -   `ThreadPoolExecutor`: Good for I/O-bound tasks (network requests, file operations where the CPU waits) or when you need simplicity in notebooks. Limited by GIL for CPU-bound Python code.
    -   `ProcessPoolExecutor`: Good for CPU-bound tasks as it uses separate processes, bypassing the GIL. Generally more robust in notebooks than `mp.Pool` but can still have issues with complex objects.
    -   `multiprocessing.Pool`: The classic library for process-based parallelism. Very powerful for CPU-bound tasks in `.py` scripts but can be trickier with pickling and the `__main__` guard in notebooks.
- **Choosing `max_workers`**:
    -   For CPU-bound tasks with `ProcessPoolExecutor` or `mp.Pool`, `os.cpu_count()` is a good starting point.
    -   For I/O-bound tasks with `ThreadPoolExecutor`, you can often use more workers than CPU cores (e.g., 2x, 4x, or even more) because threads will spend time waiting for I/O.
    -   Always benchmark to find the optimal number for your specific workload!
- **SLURM**: The `--cpus-per-task` in SLURM should ideally match the number of processes/workers your Python script intends to use for CPU-bound work.
>>>>>>> REPLACE
</diff>
</apply_diff>