# Simple build script for MkDocs

import subprocess
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Serve the site
        subprocess.run(["mkdocs", "serve"], check=True)
    else:
        # Build the site
        subprocess.run(["mkdocs", "build"], check=True)
        print("Site built successfully!")

if __name__ == "__main__":
    main() 