import os
import sys
import hashlib
import subprocess
import pytest

# Test constants
TEST_EMAIL = "test@example.com"
EXPECTED_SHA256_HASH = hashlib.sha256(TEST_EMAIL.encode()).hexdigest()

def setup_function():
    """Remove hash.email file if it exists before each test"""
    if os.path.exists("hash.email"):
        os.remove("hash.email")

def teardown_function():
    """Remove hash.email file if it exists after each test"""
    if os.path.exists("hash.email"):
        os.remove("hash.email")

def test_script_exists():
    """Test that email_hasher.py exists"""
    assert os.path.exists("email_hasher.py"), "email_hasher.py file not found"

def test_command_line_args():
    """Test that the script accepts command line arguments"""
    # Run the script with the test email
    result = subprocess.run(
        [sys.executable, "email_hasher.py", TEST_EMAIL],
        capture_output=True,
        text=True
    )
    
    # Check that the script executed without errors
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
    
    # Check that the hash.email file was created
    assert os.path.exists("hash.email"), "hash.email file was not created"

def test_sha256_algorithm():
    """Test that the script uses SHA-256 algorithm"""
    # Run the script with the test email
    subprocess.run(
        [sys.executable, "email_hasher.py", TEST_EMAIL],
        capture_output=True,
        text=True
    )
    
    # Read the hash from the file
    with open("hash.email", "r") as f:
        hash_output = f.read().strip()
    
    # Check that the hash matches the expected SHA-256 hash
    assert hash_output == EXPECTED_SHA256_HASH, "Hash does not match expected SHA-256 hash"

def test_hex_format():
    """Test that the hash is in hexadecimal format"""
    # Run the script with the test email
    subprocess.run(
        [sys.executable, "email_hasher.py", TEST_EMAIL],
        capture_output=True,
        text=True
    )
    
    # Read the hash from the file
    with open("hash.email", "r") as f:
        hash_output = f.read().strip()
    
    # Check that the hash is in hexadecimal format (64 characters for SHA-256)
    assert len(hash_output) == 64, "Hash length is not 64 characters"
    assert all(c in "0123456789abcdef" for c in hash_output.lower()), "Hash contains non-hexadecimal characters"

def test_file_creation():
    """Test that the script creates a file named 'hash.email'"""
    # Run the script with the test email
    subprocess.run(
        [sys.executable, "email_hasher.py", TEST_EMAIL],
        capture_output=True,
        text=True
    )
    
    # Check that the hash.email file was created
    assert os.path.exists("hash.email"), "hash.email file was not created"
    
    # Check that the file contains content
    with open("hash.email", "r") as f:
        content = f.read().strip()
    
    assert content, "hash.email file is empty"

def test_error_handling():
    """Test that the script handles missing command line arguments"""
    # Run the script without arguments
    result = subprocess.run(
        [sys.executable, "email_hasher.py"],
        capture_output=True,
        text=True
    )
    
    # Check that the script exits with a non-zero status
    assert result.returncode != 0, "Script should exit with error when no email is provided"
