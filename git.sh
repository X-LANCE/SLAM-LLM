#!/bin/bash

# Define the maximum file size you want to track
MAXSIZE=5M

# Find files larger than MAXSIZE, remove leading './' and append to .gitignore
find . -size +$MAXSIZE -type f | sed 's|^\./||' >> .gitignore

# Remove duplicate lines from .gitignore
sort -u -o .gitignore .gitignore