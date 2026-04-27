# PIRL: Physically Informed Reinforcement Learning

This repository contains the implementation of a **P**hysically **I**nformed **R**einforcement **L**earning agent. The project utilizes Proximal Policy Optimization (PPO) to train a robotic agent for the "Lift" manipulation task within the [Robosuite](https://robosuite.ai/) simulation environment.

The core idea is to leverage physical properties and simulated sensor data (RGB, depth, proprioception) to train a robust policy capable of performing complex tasks.

## Project Structure

The project is organized into several key Python modules:

-   **`model.py`**: Defines the `ActorCritic` neural network. It features a multimodal backbone that processes visual (RGB and depth) and physical (proprioceptive) data to create a rich, fused representation for decision-making. [1]
-   **`ppo.py`**: Contains the core implementation of the Proximal Policy Optimization (PPO) algorithm. It manages the training configuration, policy updates, and model checkpointing. [2]
-   **`ppo_train.py`**: The main executable script for launching the training process. It orchestrates the environment, the PPO agent, and the data collection loop. [3]
-   **`robotenv.py`**: Provides utilities for creating and managing single and vectorized (parallel) Robosuite environments, ensuring efficient data collection. [4]
-   **`rollout.py`**: Implements the `RolloutBuffer`, a crucial component for storing and preparing the agent's experience data (observations, actions, rewards) for policy updates. [5]
-   **`video.py`**: A utility script to render and save videos of the trained agent's performance, allowing for qualitative evaluation. [6]
-   **`pyproject.toml`**: Specifies the project's dependencies and metadata. [7]

## Getting Started

To get started with this project, follow the steps below. This guide assumes you have [uv](https://github.com/astral-sh/uv) installed, which is used for fast Python package management.

### 1. Clone the Repository

```bash
git clone https://github.com/eljandoubi/pirl.git
cd pirl
```
### 2. Set Up the Environment and Install Dependencies
This project uses uv to manage the virtual environment and dependencies. The justfile provides a convenient wrapper for this.

Run the following command to create a virtual environment and install all required packages from pyproject.toml:
```bash
just setup
```
This will create a .venv directory and install packages like torch, robosuite, and wandb.

## Usage
The justfile provides simple commands to run the main functionalities of the project.

### Training the Agent
To start the PPO training process, simply run:
```bash
just train
```
Training progress and metrics will be logged to the console and, if configured, to Weights & Biases. Model checkpoints are saved periodically in the ppo_checkpoints/ directory.

### Rendering a Video
After training, you can render a video of the best-performing agent. You need to provide the run_id from your training session.
```bash
# Replace <your_run_id> with the actual ID from your W&B run
just video <your_run_id>
```
This will load the best model from that run and save an .mp4 video file in the ppo_videos/ directory.

## Cleaning Up
To remove all generated artifacts, including the virtual environment, checkpoints, and videos, run:
```bash
just clean
```