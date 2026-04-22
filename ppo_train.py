import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import gc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm import trange  # noqa: E402

import wandb  # noqa: E402
from ppo import PPO, TrainingConfig  # noqa: E402
from robotenv import SubprocVecEnv, make_env  # noqa: E402
from rollout import RolloutBuffer  # noqa: E402

print("Loading environment variables...", load_dotenv())

def main():
    exit_code = 0
    try:
        ############## Hyperparameters ##############
        config = TrainingConfig()
        run = wandb.init(project="ppo-robosuite", config=config.__dict__, id=config.runid, resume="allow")
        config.update_path(run.id)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _env = make_env(img_size=config.img_size)
        action_dim = _env.action_dim
        keys = ["agentview_image", "agentview_depth", "robot0_proprio-state"]
        obs_shapes = {k: _env.observation_spec()[k].shape for k in keys}
        _env.close()
        del _env

        env_kwargs = dict(img_size=config.img_size,
        max_episode_steps=config.max_ep_len,)
    
        env = SubprocVecEnv(
        make_env,
        num_envs=config.num_envs,
        env_kwargs=env_kwargs,
        filter_keys=keys,
    )


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
                obs = env.reset()
                current_ep_reward = 0
                for t in range(1, config.max_ep_len + 1):
                    
                    # select action with policy
                    action, log_prob, values, obs = ppo_agent.select_action(obs)
                    env_actions = action.detach().cpu().numpy().clip(-1, 1)
                    next_obs, rewards, dones = env.step(env_actions)
                    buffer.add(obs, action, rewards, dones, values, log_prob)
                    obs = next_obs
                    time_step += 1
                    current_ep_reward += np.mean(rewards)

                    # update PPO agent
                    if len(buffer) >= config.update_timestep:
                        assert len(buffer) == config.update_timestep, f"Buffer length {len(buffer)} does not match expected {config.update_timestep}"
                        ppo_agent.update(buffer, next_obs)
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
                pbar.set_postfix({"Timestep": time_step, "Reward": current_ep_reward})

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

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     exit_code = 1

    finally:

        wandb.finish(exit_code=exit_code)
        env.close()


if __name__ == "__main__":
    main()
