import gc
import os

from rollout import RolloutBuffer

os.environ["MUJOCO_GL"] = "osmesa"  # "egl" #

import numpy as np
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from tqdm import trange

import wandb
from ppo import PPO, TrainingConfig
from robotenv import VecEnv, make_env

print("Loading environment variables...", load_dotenv())

def main():
    mp.set_start_method("spawn", force=True)
    ############## Hyperparameters ##############
    config = TrainingConfig()
    run = wandb.init(project="ppo-robosuite", config=config.__dict__, id=config.runid, resume="allow")
    config.update_path(run.id)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_env(img_size=config.img_size)
    action_dim = env.action_dim
    keys = ["agentview_image", "agentview_depth", "robot0_proprio-state"]
    obs_shapes = {k: env.observation_spec()[k].shape for k in keys}
    env.close()
    del env
    envs = VecEnv(make_env, config.num_envs, img_size=config.img_size,
                  max_episode_steps=config.max_ep_len)

    buffer = RolloutBuffer(config.max_ep_len, config.num_envs, obs_shapes, action_dim, device)
    print(buffer)
    ppo_agent = PPO(
        action_dim,
        device,
        obs_shapes,
        config
    )

    if config.load_checkpoint_path:
        ppo_agent.load(config.load_checkpoint_path)

    time_step = 0

    num_episodes = int(
        config.max_training_timesteps // config.update_timestep
    )+1
    with trange(num_episodes, desc="Episodes") as pbar:
        for ep in range(num_episodes):
            states = envs.reset()
            current_ep_reward = 0
            for t in range(1, config.max_ep_len + 1):
                # select action with policy
                action, log_prob, values, obs = ppo_agent.select_action(states)
                envs_actions = action.detach().cpu().numpy().clip(-1, 1)
                next_states, rewards, dones = envs.step(envs_actions)
                buffer.add(obs, action, rewards, dones, values, log_prob)
                states = next_states
                time_step += 1
                current_ep_reward += np.mean(rewards)

                # update PPO agent
                if len(buffer) >= config.update_timestep:
                    assert len(buffer) == config.update_timestep, f"Buffer length {len(buffer)} does not match expected {config.update_timestep}"
                    ppo_agent.update(buffer, next_states)
                    buffer.reset()
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

            # Log to wandb
            log_payload = {
                "episode": ep,
                "reward": current_ep_reward,
            }
            if ppo_agent.mse_losses:
                log_payload["mse_loss"] = ppo_agent.mse_losses[-1]
            if ppo_agent.entropy_losses:
                log_payload["entropy_loss"] = ppo_agent.entropy_losses[-1]
            if ppo_agent.surrogate_losses:
                log_payload["surrogate_loss"] = ppo_agent.surrogate_losses[-1]
            
            wandb.log(
                log_payload, step=time_step
            )

            pbar.set_postfix({"Timestep": time_step, "Reward": current_ep_reward})

    envs.close()
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
