#!/usr/bin/env python3
"""
Email Hasher Script

This script takes an email address as a command line argument,
hashes it using the SHA-256 algorithm, and prints the hash to stdout.

Usage:
    python email_hasher.py <email_address>

Example:
    python email_hasher.py example@email.com
"""

import sys
import hashlib

def hash_email(email):
    """
    Hash an email address using SHA-256 and return the hexadecimal digest.
    
    Args:
        email (str): The email address to hash
        
    Returns:
        str: The SHA-256 hash of the email in hexadecimal format
    """
    # TODO: Implement this function
    # 1. Convert the email string to bytes
    # 2. Create a SHA-256 hash of the email
    # 3. Return the hash in hexadecimal format
    pass

def main():
    """
    Main function to process command line arguments and execute the script.
    """
    # TODO: Implement this function
    # 1. Check if an email address was provided as a command line argument
    # 2. If not, print an error message and exit with a non-zero status
    # 3. If yes, hash the email address
    # 4. Print the hash to stdout
    pass

if __name__ == "__main__":
    main()
