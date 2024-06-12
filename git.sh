#!/bin/bash

# Define the maximum file size you want to track
MAXSIZE=100M

# Find files larger than MAXSIZE and append them to .gitignore
find . -size +$MAXSIZE -type f | sed 's|^\./||' >> .gitignore