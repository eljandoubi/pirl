import gc
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
# Logging and plotting
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange

from ppo import PPO, Memory
from robotenv import VecEnv, make_env


def main():
    mp.set_start_method("spawn", force=True)
    ############## Hyperparameters ##############
    env_name = "robosuite_lift"
    max_ep_len = 200  # max timesteps in one episode
    max_training_timesteps = (
        1000000  # break training loop if timeteps > max_training_timesteps
    )

    K_epochs = 10
    update_timestep = 4000
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 1e-4
    lr_critic = 3e-4

    img_size = 64  # Image size for CNN input
    num_envs = 4  # Number of parallel environments

    # --- Checkpointing ---
    save_model_freq = int(2e4)  # Save model every n timesteps
    checkpoint_dir = "./ppo_checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # To resume training, set `load_checkpoint_path` to the model file
    # e.g., load_checkpoint_path = "./ppo_checkpoints/PPO_robosuite_lift_20000.pth"
    load_checkpoint_path = None
    #############################################

    # Logging
    log_dir = "./ppo_logs"
    os.makedirs(log_dir, exist_ok=True)
    reward_log_path = os.path.join(log_dir, "rewards.txt")
    loss_log_path = os.path.join(log_dir, "losses.txt")
    reward_history = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_env(img_size=img_size)
    action_dim = env.action_dim
    proprio_dim = env.observation_spec()["robot0_proprio-state"].shape[0]
    del env
    envs = VecEnv(make_env, num_envs, img_size=img_size)

    memory = Memory(device)
    ppo_agent = PPO(action_dim, img_size, proprio_dim, lr_actor, lr_critic, 
                    gamma, K_epochs, eps_clip, device)

    if load_checkpoint_path:
        ppo_agent.load(load_checkpoint_path)

    time_step = 0


    num_episodes = int(max_training_timesteps // max_ep_len)
    with trange(num_episodes, desc="Episodes") as pbar:
        for _ in range(num_episodes):
            states = envs.reset()
            current_ep_reward = 0
            for t in range(1, max_ep_len + 1):
                # select action with policy
                action, log_prob = ppo_agent.select_action(states)
                next_states, rewards, dones = envs.step(action.numpy())
                memory.extend(action, states, log_prob, rewards, dones)
                states = next_states
                time_step += 1
                current_ep_reward += np.mean(rewards)

                # update PPO agent
                if len(memory) >= update_timestep:
                    ppo_agent.update(memory)
                    memory.clear_memory()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # save model checkpoint
                if time_step % save_model_freq == 0:
                    checkpoint_path = f"{checkpoint_dir}/PPO_{env_name}_{time_step}.pth"
                    ppo_agent.save(checkpoint_path)
            
            pbar.update(1)
            # Logging reward
            reward_history.append(current_ep_reward)
            with open(reward_log_path, "a") as f:
                f.write(f"{current_ep_reward}\n")

            pbar.set_postfix({"Timestep": time_step, "Reward": current_ep_reward})

    envs.close()

    # Save losses to file
    with open(loss_log_path, "w") as f:
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
    plt.savefig(os.path.join(log_dir, "training_curves.png"))
    plt.show()


if __name__ == "__main__":
    main()
