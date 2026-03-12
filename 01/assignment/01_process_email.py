#!/usr/bin/env python3
"""Process UCSF email and create verification hash."""

import sys
import hashlib
import re

def process_email(email_address):
    """Extract username, clean it, and create hash."""
    username = email_address.split('@')[0]
    username_clean = re.sub(r'[^a-z0-9]', '', username.lower().strip())
    hash_object = hashlib.sha256(username_clean.encode())
    return hash_object.hexdigest()

def main():
    if len(sys.argv) != 2:
        print("Usage: python 01_process_email.py your.email@ucsf.edu")
        sys.exit(1)

    email = sys.argv[1]
    hash_value = process_email(email)

    with open("processed_email.txt", "w") as f:
        f.write(f"{hash_value}\n")

    print(f"✓ Email processed successfully!")
    print(f"Hash saved to processed_email.txt")

if __name__ == "__main__":
    main()
