import sqlite3
import pytest
import os

# Define the expected success message
# Note: This might need adjustment if the database message changes slightly
MURDER_SUCCESS = "Congrats, you found the murderer!"
MASTERMIND_SUCCESS = "Congrats, you found the brains behind the murder!"

# Define the path to the database relative to the test script
# Assumes the test runs from the repository root or .github directory context
DATABASE_FILE = "sql-murder-mystery.db"
# Adjust path if running from .github/tests
if not os.path.exists(DATABASE_FILE) and os.path.exists(os.path.join("..", "..", DATABASE_FILE)):
    DATABASE_FILE = os.path.join("..", "..", DATABASE_FILE)


def get_solution_value():
    """Connects to the database and retrieves the value from the solution table."""
    if not os.path.exists(DATABASE_FILE):
        pytest.fail(f"Database file not found at expected path: {os.path.abspath(DATABASE_FILE)}")

    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM solution LIMIT 1;")
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        pytest.fail(f"Database error: {e}")
    finally:
        if conn:
            conn.close()


def test_murder_mystery_solution():
    """Tests if the solution table contains the correct success message."""
    solution_value = get_solution_value()

    assert solution_value is not None, "No value found in the solution table. Did the notebook run correctly and insert the solution?"
    assert isinstance(solution_value, str), f"Expected solution value to be a string, but got {type(solution_value)}"
    assert solution_value.strip().startswith(MURDER_SUCCESS) or solution_value.strip().startswith(MASTERMIND_SUCCESS), \
        f"Solution value '{solution_value}' does not start with the expected success message. Try again!"