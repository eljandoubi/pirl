import gc
import os

os.environ["MUJOCO_GL"] = "osmesa"  # "egl" #
# Logging and plotting

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from tqdm import trange

import wandb
from ppo import PPO, Memory, TrainingConfig
from robotenv import VecEnv, make_env

print("Loading environment variables...", load_dotenv())

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
    envs = VecEnv(make_env, config.num_envs, img_size=config.img_size,
                  max_episode_steps=config.max_ep_len)

    memory = Memory(device)
    ppo_agent = PPO(
        action_dim,
        proprio_dim,
        device,
        config
    )

    if config.load_checkpoint_path:
        ppo_agent.load(config.load_checkpoint_path)

    time_step = 0

    num_episodes = int(
        config.max_training_timesteps // (config.max_ep_len * config.num_envs)
    )
    with trange(num_episodes, desc="Episodes") as pbar:
        for ep in range(num_episodes):
            states = envs.reset()
            current_ep_reward = 0
            for t in range(1, config.max_ep_len + 1):
                # select action with policy
                action, log_prob = ppo_agent.select_action(states)
                next_states, rewards, dones = envs.step(action.numpy().clip(-1, 1))
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
                    checkpoint_path = (
                        f"{config.checkpoint_dir}/PPO_{config.env_name}_{time_step}.pth"
                    )
                    ppo_agent.save(checkpoint_path)

            pbar.update(1)
            # Logging reward
            reward_history.append(current_ep_reward)
            with open(config.reward_log_path, "a") as f:
                f.write(f"{current_ep_reward}\n")

            # Log to wandb
            wandb.log(
                {
                    "episode": ep,
                    "reward": current_ep_reward,
                    "timestep": time_step,
                    "loss": ppo_agent.losses[-1] if ppo_agent.losses else None,
                }
            )

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
