#!/bin/bash

# Install Playwright dependencies
playwright install chromium

# Build the site
mkdocs build 