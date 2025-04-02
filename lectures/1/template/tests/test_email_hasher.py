import os
import sys
import hashlib
import subprocess
import pytest

# Test constants
TEST_INPUT = "42"
EXPECTED_SHA256_HASH = hashlib.sha256(TEST_INPUT.encode()).hexdigest()

def test_script_exists():
    """Test that email_hasher.py exists"""
    assert os.path.exists("email_hasher.py"), "email_hasher.py file not found"

def test_correct_hash():
    """Test that the script returns the correct hash for a known input"""
    result = subprocess.run(
        [sys.executable, "email_hasher.py", TEST_INPUT],
        capture_output=True,
        text=True
    )
    
    assert result.stdout.strip() == EXPECTED_SHA256_HASH, "Hash does not match expected SHA-256 hash"

def test_hash_file_format():
    """Test that hash.email contains something that looks like a hash"""
    assert os.path.exists("hash.email"), "hash.email file was not created"
    
    with open("hash.email", "r") as f:
        content = f.read().strip()
    
    # Check that it looks like a hash (64 hex chars)
    assert len(content) == 64, "Hash length is not 64 characters"
    assert all(c in "0123456789abcdef" for c in content.lower()), "Hash contains non-hexadecimal characters"
