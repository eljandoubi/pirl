import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from model import ActorCritic
from rollout import RolloutBuffer


@dataclass
class TrainingConfig:
    env_name: str = "robosuite_lift"
    max_ep_len: int = 1024  # max timesteps in one episode
    max_training_timesteps: int = (
        1000000  # break training loop if timeteps > max_training_timesteps
    )

    K_epochs: int = 100
    eps_clip: float = 0.2  # clip parameter for PPO
    gamma: float = 0.99  # discount factor
    lam: float = 0.95  # GAE lambda parameter

    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    entropy_coef: float = 0.0005
    mse_coef: float = 0.5
    action_std: float = 0.5  # Standard deviation for action distribution (if fixed variance)
    fixed_policy_variance: bool = True  # Whether to use a fixed variance for the action distribution
    max_grad_norm: float = 1.  # Max gradient norm for clipping
    img_size: int = 64  # Image size for CNN input
    num_envs: int = 8  # Number of parallel environments

    # --- Checkpointing ---
    checkpoint_dir: str = "./ppo_checkpoints"
    load_checkpoint_path: str | None = None

    runid: str | None = None  # Wandb run ID for resuming runs

    def __post_init__(self):
        
        self.update_timestep = self.max_ep_len * self.num_envs  # Number of timesteps to collect before each PPO update
        assert self.lr_actor > 0, "Learning rate for actor must be positive"
        assert self.lr_critic > 0, "Learning rate for critic must be positive"
        assert self.gamma > 0 and self.gamma <= 1, "Gamma must be in (0, 1]"
        assert self.lam >= 0 and self.lam <= 1, "Lambda must be in [0, 1]"
        assert self.eps_clip > 0, "Epsilon clip must be positive"
        assert self.K_epochs > 0, "K_epochs must be positive"
        assert self.max_ep_len > 0, "max_ep_len must be positive"
        assert self.num_envs > 0, "num_envs must be positive"
        if self.load_checkpoint_path is not None:
            assert os.path.isfile(self.load_checkpoint_path), f"Checkpoint path {self.load_checkpoint_path} does not exist"
        if self.fixed_policy_variance:
            assert self.action_std > 0, "Action standard deviation must be positive when using fixed policy variance"
            assert self.action_std <= 1, "Action standard deviation should be less than 1 for stable training"

    def update_path(self, folder_name: str | None = None):
        if folder_name is None:
            folder_name = self.runid if self.runid is not None else "default_run"
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, folder_name)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def set_id(self, runid: str):
        self.runid = runid


class PPO:
    def __init__(
        self, action_dim:int, device:torch.device, 
        obs_shapes: dict[str,tuple[int,...]], config: TrainingConfig

    ):
        self.config = config
        self.device = device
        self.obs_shapes = obs_shapes

        self.policy = ActorCritic(action_dim, config.img_size, proprio_dim=obs_shapes["robot0_proprio-state"][0],
                                  fixed_policy_variance=config.fixed_policy_variance, action_std=config.action_std
                                  ).to(device).train()
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy.actor.parameters(), "lr": config.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": config.lr_critic},
            ]
        )

        self.policy_old = ActorCritic(action_dim, config.img_size,  proprio_dim=obs_shapes["robot0_proprio-state"][0],
                                      fixed_policy_variance=config.fixed_policy_variance, action_std=config.action_std
                                      ).to(device).eval()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # For logging losses
        self.mse_losses = []
        self.entropy_losses = []
        self.surrogate_losses = []


    @staticmethod
    def obs_to_device(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device, non_blocking=True) for k, v in obs.items()}

    @torch.no_grad()
    def select_action(self, obs: dict[str, torch.Tensor]):
        obs = self.obs_to_device(obs, self.device)
        action_mean, action_var, values = self.policy_old(obs)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob, values, obs
    


    def update(self, buffer: RolloutBuffer, last_obs: dict[str, torch.Tensor]):
        last_obs = self.obs_to_device(last_obs, self.device)
        with torch.no_grad():
            last_value = self.policy(last_obs)[2].squeeze()

        returns, advantages = buffer.compute_gae(last_value, self.config.gamma, self.config.lam)
        
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch = buffer.get()

        # Optimize policy for K epochs
        epoch_mse_losses = []
        epoch_entropy_losses = []
        epoch_surrogate_losses = []
        eps_clip = self.config.eps_clip
        for _ in tqdm(range(self.config.K_epochs), desc="PPO Update Epochs"):
            # Evaluating old actions and values
            action_mean, action_var, state_values = self.policy(batch["obs"])
            
            mse_loss = self.MseLoss(state_values.squeeze(1), returns)
            epoch_mse_losses.append(mse_loss.item())

            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            dist_entropy = dist.entropy().mean()
            epoch_entropy_losses.append(dist_entropy.item())


            logprobs = dist.log_prob(batch["actions"])
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - batch["logprobs"].detach())
            # Finding Surrogate Loss
            
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            )
            surrgate = -torch.min(surr1, surr2).mean()
            epoch_surrogate_losses.append(surrgate.item())
            # final loss of clipped objective PPO
            loss = (
                surrgate
                + self.config.mse_coef * mse_loss
                - self.config.entropy_coef * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            

        # Log average loss for this update
        avg_mse_loss = np.mean(epoch_mse_losses)
        avg_entropy_loss = np.mean(epoch_entropy_losses)
        avg_surrogate_loss = np.mean(epoch_surrogate_losses)
        self.mse_losses.append(avg_mse_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.surrogate_losses.append(avg_surrogate_loss)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path: str):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"Checkpoint loaded from {checkpoint_path}")
