import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from model import ActorCritic


@dataclass
class TrainingConfig:
    env_name: str = "robosuite_lift"
    max_ep_len: int = 200  # max timesteps in one episode
    max_training_timesteps: int = (
        1000000  # break training loop if timeteps > max_training_timesteps
    )

    K_epochs: int = 100
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

    fixed_policy_variance: bool = True  # Whether to use a fixed variance for the action distribution
    max_grad_norm: float = 1.  # Max gradient norm for clipping

    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.reward_log_path = os.path.join(self.log_dir, self.reward_log_path)
        self.loss_log_path = os.path.join(self.log_dir, self.loss_log_path)

class Memory:
    def __init__(self,device:torch.device):
        self.device = device
        self.actions: list[torch.Tensor] = []
        self.states: list[dict[str, torch.Tensor]] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.is_terminals: list[torch.Tensor] = []

    def extend(self, actions: torch.Tensor, states: list[dict[str, np.ndarray]], logprobs: torch.Tensor,
               rewards: list[float], is_terminals: list[bool]):
        states_on_device = []
        for state in states:
            state_on_device = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in state.items()
            }
            states_on_device.append(state_on_device)

        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        is_terminals = torch.as_tensor(is_terminals, dtype=torch.float32)

        self.actions.extend(actions)
        self.states.extend(states_on_device)
        self.logprobs.extend(logprobs)
        self.rewards.extend(rewards)
        self.is_terminals.extend(is_terminals)

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return {
            "action": self.actions[idx],
            "state": self.states[idx],
            "logprob": self.logprobs[idx],
            "reward": self.rewards[idx],
            "is_terminal": self.is_terminals[idx],
        }
    
    def to_tensor(self, gamma:float)->tuple[torch.Tensor,...]:
        rewards = torch.stack(self.rewards).to(self.device, non_blocking=True)
        is_terminals = torch.stack(self.is_terminals).to(self.device, non_blocking=True)
        actions = torch.stack(self.actions).to(self.device, non_blocking=True)
        states_img = torch.stack([s["agentview_image"] for s in self.states]).to(self.device, non_blocking=True)
        states_depth = torch.stack([s["agentview_depth"] for s in self.states]).to(self.device, non_blocking=True)
        states_proprio = torch.stack([s["robot0_proprio-state"] for s in self.states]).to(self.device, non_blocking=True)
        logprobs = torch.stack(self.logprobs).to(self.device, non_blocking=True)
        
        rewards = self.compute_returns(rewards, is_terminals, gamma)

        return actions, states_img, states_depth, states_proprio, logprobs, rewards
    
    @staticmethod
    def compute_returns(rewards: torch.Tensor, terminals: torch.Tensor, gamma: float):
        # Create discount factors with reset
        gamma_mask = gamma * (1 - terminals)
        gamma_mask[-1] = gamma  # edge case

        # Reverse
        rewards_rev = torch.flip(rewards, [0])
        gamma_mask_rev = torch.flip(gamma_mask, [0])

        # Cumulative product of discounts
        discounts = torch.cumprod(
            torch.cat([torch.ones(1, device=rewards.device, dtype=rewards.dtype),
                       gamma_mask_rev[:-1]]), dim=0
        )

        # Discounted cumulative sum
        discounted = torch.cumsum(rewards_rev * discounts, dim=0) / discounts

        # Flip back
        rewards = torch.flip(discounted, [0])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        return rewards
    
    def shape(self):
        res = {"len": len(self)}
        if len(self) > 0:
            res["actions"] = self.actions[0].shape
            res["states_img"] = self.states[0]["agentview_image"].shape
            res["states_depth"] = self.states[0]["agentview_depth"].shape
            res["states_proprio"] = self.states[0]["robot0_proprio-state"].shape
            res["logprobs"] = self.logprobs[0].shape,
            res["rewards"] = self.rewards[0].shape,
            res["is_terminals"] = self.is_terminals[0].shape,
        return res

class PPO:
    def __init__(
        self, action_dim:int, proprio_dim:int, device:torch.device, config: TrainingConfig

    ):
        self.config = config
        self.device = device

        self.policy = ActorCritic(action_dim, config.img_size, proprio_dim,
                                  fixed_policy_variance=config.fixed_policy_variance).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": config.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": config.lr_critic},
            ]
        )

        self.policy_old = ActorCritic(action_dim, config.img_size, proprio_dim,
                                      fixed_policy_variance=config.fixed_policy_variance).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # For logging losses
        self.losses = []

    @torch.no_grad()
    def select_action(self, state):
        # Preprocess state for the network
        if not isinstance(state, (list, tuple)):
            state = [state]
        
        img = torch.stack([torch.as_tensor(s["agentview_image"], dtype=torch.float32) for s in state]).to(
            self.device, non_blocking=True)
        depth = torch.stack([torch.as_tensor(s["agentview_depth"], dtype=torch.float32) for s in state]).to(
            self.device, non_blocking=True)
        proprio = torch.stack([torch.as_tensor(s["robot0_proprio-state"], dtype=torch.float32) for s in state]).to(
            self.device, non_blocking=True)

        obs = {
            "agentview_image": img,
            "agentview_depth": depth,
            "robot0_proprio-state": proprio,
        }

        action_mean, action_var, _ = self.policy_old(obs)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.cpu(), action_logprob.cpu()

    def update(self, memory: Memory):


        (old_actions, old_states_img, old_states_depth, old_states_proprio, 
         old_logprobs, rewards) = memory.to_tensor(self.config.gamma)

        old_obs = {
            "agentview_image": old_states_img,
            "agentview_depth": old_states_depth,
            "robot0_proprio-state": old_states_proprio,
        }

        # Optimize policy for K epochs
        epoch_losses = []
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_mean, action_var, state_values = self.policy(old_obs)
            cov_mat = torch.diag(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards.unsqueeze(1))
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            epoch_losses.append(loss.mean().item())

        # Log average loss for this update
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.losses.append(avg_loss)

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
