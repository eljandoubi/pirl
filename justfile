# Sets the shell to be used for executing recipes.
set shell := ["bash", "-e", "-u", "-o", "pipefail"]

# Default command: lists all available commands in this Justfile.
default:
    just --list

# Sets up the project by creating a virtual environment and installing dependencies.
setup:
    echo "Creating virtual environment using uv..."
    uv venv
    echo "Syncing dependencies from pyproject.toml..."
    uv sync
    echo "Setup complete. Activate the environment with 'source .venv/bin/activate'"

# Runs the PPO training script.
train:
    echo "Starting PPO training..."
    uv run pirl/ppo_train.py

# Renders a video of a trained agent's performance.
# Usage: just video <run_id>
video run_id:
    echo "Rendering video for run ID: {{run_id}}..."
    uv run pirl/video.py --runid {{run_id}}

# Cleans up the project directory by removing generated files and the virtual environment.
clean:
    echo "Cleaning up generated directories and the .venv..."
    rm -rf ppo_checkpoints
    rm -rf ppo_videos
    rm -rf .venv