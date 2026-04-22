import gc
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
# Logging and plotting
from dataclasses import dataclass

import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange

from ppo import PPO, Memory
from robotenv import VecEnv, make_env


@dataclass
class TrainingConfig:
    env_name: str = "robosuite_lift"
    max_ep_len: int = 200  # max timesteps in one episode
    max_training_timesteps: int = 1000000  # break training loop if timeteps > max_training_timesteps

    K_epochs: int = 10
    update_timestep: int = 4000
    eps_clip: float = 0.2  # clip parameter for PPO
    gamma: float = 0.99  # discount factor

    lr_actor: float = 1e-4
    lr_critic: float = 3e-4

    img_size: int = 64  # Image size for CNN input
    num_envs: int = 10  # Number of parallel environments

    # --- Checkpointing ---
    save_model_freq: int = int(2e4)  # Save model every n timesteps
    checkpoint_dir: str = "./ppo_checkpoints"
    load_checkpoint_path: str | None = None
    log_dir: str = "./ppo_logs"
    reward_log_path: str = "rewards.txt"
    loss_log_path: str = "losses.txt"



    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.reward_log_path = os.path.join(self.log_dir, self.reward_log_path)
        self.loss_log_path = os.path.join(self.log_dir, self.loss_log_path)


def main():
    mp.set_start_method("spawn", force=True)
    ############## Hyperparameters ##############
    config = TrainingConfig()
    reward_history = []

    # Initialize wandb
    wandb.init(project="ppo-robosuite", config=config.__dict__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_env(img_size=config.img_size)
    action_dim = env.action_dim
    proprio_dim = env.observation_spec()["robot0_proprio-state"].shape[0]
    del env
    envs = VecEnv(make_env, config.num_envs, img_size=config.img_size)

    memory = Memory(device)
    ppo_agent = PPO(action_dim, config.img_size, proprio_dim, config.lr_actor, config.lr_critic,
                    config.gamma, config.K_epochs, config.eps_clip, device)

    if config.load_checkpoint_path:
        ppo_agent.load(config.load_checkpoint_path)

    time_step = 0


    num_episodes = int(config.max_training_timesteps // (config.max_ep_len * config.num_envs))
    with trange(num_episodes, desc="Episodes") as pbar:
        for ep in range(num_episodes):
            states = envs.reset()
            current_ep_reward = 0
            for t in range(1, config.max_ep_len + 1):
                # select action with policy
                action, log_prob = ppo_agent.select_action(states)
                next_states, rewards, dones = envs.step(action.numpy())
                memory.extend(action, states, log_prob, rewards, dones)
                states = next_states
                time_step += 1
                current_ep_reward += np.mean(rewards)

                # update PPO agent
                if len(memory) >= config.update_timestep:
                    ppo_agent.update(memory)
                    memory.clear_memory()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # save model checkpoint
                if time_step % config.save_model_freq == 0:
                    checkpoint_path = f"{config.checkpoint_dir}/PPO_{config.env_name}_{time_step}.pth"
                    ppo_agent.save(checkpoint_path)

            pbar.update(1)
            # Logging reward
            reward_history.append(current_ep_reward)
            with open(config.reward_log_path, "a") as f:
                f.write(f"{current_ep_reward}\n")

            # Log to wandb
            wandb.log({
                "episode": ep,
                "reward": current_ep_reward,
                "timestep": time_step,
                "loss": ppo_agent.losses[-1] if ppo_agent.losses else None
            })

            pbar.set_postfix({"Timestep": time_step, "Reward": current_ep_reward})

    envs.close()

    # Save losses to file
    with open(config.loss_log_path, "w") as f:
        for loss in ppo_agent.losses:
            f.write(f"{loss}\n")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(reward_history)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(ppo_agent.losses)
    plt.title("PPO Loss (per update)")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, "training_curves.png"))
    plt.show()

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
