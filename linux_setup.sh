#!/bin/bash

# Store the current directory path
current_dir="$(pwd)"

# Define the project directory path
project_dir="$current_dir/Dough"

# Check if the "Dough" directory doesn't exist and we're not already inside it
if [ ! -d "$project_dir" ] && [ "$(basename "$current_dir")" != "Dough" ]; then
    # Clone the git repo
    git clone --depth 1 -b main https://github.com/banodoco/Dough.git "$project_dir"
    cd "$project_dir"
    git clone --depth 1 -b feature/package https://github.com/piyushK52/comfy_runner.git

    # Create virtual environment
    python3 -m venv "dough-env"
    
    # Install system dependencies
    if command -v sudo &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y libpq-dev python3.10-dev
    else
        apt-get update && apt-get install -y libpq-dev python3.10-dev
    fi

    # Install Python dependencies
    echo $(pwd)
    . ./dough-env/bin/activate && pip install -r "requirements.txt"
    . ./dough-env/bin/activate && pip install -r "comfy_runner/requirements.txt"

    # Copy the environment file
    cp "$project_dir/.env.sample" "$project_dir/.env"
fi
