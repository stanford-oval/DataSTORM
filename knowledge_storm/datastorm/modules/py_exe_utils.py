import json
import subprocess
import os
import pathlib

SQL_RESULTS_DIR = os.getenv(
    "DATASTORM_SQL_RESULTS_DIR",
    str(pathlib.Path(__file__).resolve().parents[3] / "sql_results"),
)
import tempfile
import uuid
import re


def execute_python_code_in_sandbox(code: str, timeout: int = 60) -> dict:
    """
    Executes the given Python `code` string in a throwaway Docker container to help mitigate malicious commands.

    Returns:
      - dict with { "stdout": ..., "stderr": ..., "returncode": ... }
      or { "error": ... } in case of any timeout or unexpected exceptions.
    """
    # Docker image that has Python installed -- can pin a version, e.g. "python:3.9-alpine"
    docker_image = "python:3.10"
    
    # Make a unique filename
    unique_id = uuid.uuid4().hex
    filename = f"/tmp/script_{unique_id}.py"
    
    # Write code to a local temp file so we can mount it into the Docker container.
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".py") as tmpf:
        tmpf.write(code)
        tmpf.flush()
        local_path = tmpf.name

    try:
        # Run the Docker container, mounting the temp file to /sandbox/script.py inside the container.
        # --rm : remove container on exit
        # -v local_path:/sandbox/script.py:ro : read-only mount to reduce potential tampering
        # --network none : no network access in container, further limiting malicious ops
        # --memory, --cpus (if your Docker supports it) could also be used for resource limits.
        # This container only runs the code in an isolated environment and then shuts down.

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{SQL_RESULTS_DIR}/:{SQL_RESULTS_DIR}/",
            "-v", f"{local_path}:/sandbox_script.py:ro",
            "-w", "/sandbox",
            docker_image,
            "sh", "-c",
            # "pip install pandas==2.2.3 && pip install statsmodels && pip install scikit-learn && pip install -q matplotlib && pip install -q seaborn && python3 /sandbox_script.py"
            "pip install pandas==2.2.3 && pip install -q plotly && python3 /sandbox_script.py"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        # Clean stdout to remove pip installation output
        stdout_clean = ""
        if "threadpoolctl" in result.stdout:
            try:
                stdout_clean = result.stdout[result.stdout.index("threadpoolctl-3.6.0\n")+len("threadpoolctl-3.6.0\n"):].strip()
            except ValueError:
                stdout_clean = result.stdout
        else:
            stdout_clean = result.stdout
        
        # Filter stderr to remove repetitive pip warnings
        stderr_clean = ""
        if result.stderr:
            # Remove pip warning messages that repeat
            pip_warnings = [
                r".*WARNING: Running pip as the 'root' user.*",
                r".*\[notice\] A new release of pip is available.*",
                r".*\[notice\] To update, run: pip install --upgrade pip"
            ]
            stderr_lines = result.stderr.splitlines()
            unique_stderr_lines = []
            
            # Skip all lines that match any of the warning patterns
            for line in stderr_lines:
                matched = False
                for pattern in pip_warnings:
                    if re.match(pattern, line):
                        matched = True
                        break
                if not matched:
                    unique_stderr_lines.append(line)
            
            stderr_clean = "\n".join(unique_stderr_lines)
        
        return {
            "stdout": stdout_clean,
            "stderr": stderr_clean,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup the local temporary file
        try:
            os.remove(local_path)
        except OSError:
            pass


def execute_python_script(python_script: str) -> dict:
    """
    A function that simulates the node's logic in your pipeline.
    Returns a dictionary with "stdout", "stderr", "returncode", or "error".
    """
    result = execute_python_code_in_sandbox(python_script, timeout=60)
    return result


if __name__ == "__main__":
    code_to_run = r'''
# Question: Is there a significant correlation between the priority level and resolution time of incidents across all categories?

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the CSV file into a DataFrame
file_path = os.path.join(SQL_RESULTS_DIR, "example.csv")
df = pd.read_csv(file_path)

# Assuming the CSV contains a 'resolution_time' column (in hours, days, etc.)
# and 'priority' column is categorical, we need to encode priority levels numerically.
priority_mapping = {
    "1 - Critical": 1,
    "2 - High": 2,
    "3 - Moderate": 3,
    "4 - Low": 4
}
df['priority_numeric'] = df['priority'].map(priority_mapping)

# Check for missing values in relevant columns
if df[['priority_numeric', 'resolution_time']].isnull().any().any():
    df = df.dropna(subset=['priority_numeric', 'resolution_time'])

# Calculate the Spearman correlation between priority and resolution time
correlation, p_value = spearmanr(df['priority_numeric'], df['resolution_time'])

# Print the correlation result
print(f"Spearman Correlation: {correlation}")
print(f"P-value: {p_value}")

# Visualize the relationship using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='priority_numeric', y='resolution_time', data=df, alpha=0.6)
plt.title('Priority Level vs Resolution Time')
plt.xlabel('Priority Level (Numeric)')
plt.ylabel('Resolution Time')
plt.xticks(ticks=[1, 2, 3, 4], labels=["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"])
plt.grid(True)
plt.show()
'''

    # Call our function
    print("=== Testing execute_python_script ===")
    result_dict = execute_python_script(code_to_run)
    
    # Display the results
    print("Result dictionary:", json.dumps(result_dict, indent=2))
    
    # If you want just stdout
    stdout_str = result_dict.get("stdout", "")
    print("\n=== STDOUT ===\n", stdout_str)
    stderr_str = result_dict.get("stderr", "")
    print("\n=== STDERR ===\n", stderr_str)
    
    # If there was an error, show it
    if "error" in result_dict:
        print("\n=== ERROR ===\n", result_dict["error"])