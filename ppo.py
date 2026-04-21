import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from model import ActorCritic


# --- 3. PPO Agent ---
class PPO:
    def __init__(
        self, action_dim,img_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(action_dim, img_size).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(action_dim, img_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.action_var = torch.full((action_dim,), 0.5**2).to(device)

        # For logging losses
        self.losses = []

    def select_action(self, state):
        with torch.no_grad():
            # Preprocess state for the network
            img = (
                torch.tensor(state["agentview_image"], dtype=torch.float32)
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            depth = (
                torch.tensor(state["agentview_depth"], dtype=torch.float32)
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            proprio = (
                torch.tensor(state["robot0_proprio-state"], dtype=torch.float32)
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )

            obs = {
                "agentview_image": img,
                "agentview_depth": depth,
                "robot0_proprio-state": proprio,
            }

            action_mean, _ = self.policy_old(obs)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)

            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.detach().cpu().numpy().flatten(), action_logprob.detach()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device, non_blocking=True)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states_img = torch.stack([s["agentview_image"] for s in memory.states]).to(self.device, non_blocking=True)
        old_states_depth = torch.stack([s["agentview_depth"] for s in memory.states]).to(self.device, non_blocking=True)
        old_states_proprio = torch.stack([s["robot0_proprio-state"] for s in memory.states]).to(self.device, non_blocking=True)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).to(self.device, non_blocking=True)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).to(
            self.device, non_blocking=True
        )

        old_obs = {
            "agentview_image": old_states_img,
            "agentview_depth": old_states_depth,
            "robot0_proprio-state": old_states_proprio,
        }

        # Optimize policy for K epochs
        epoch_losses = []
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            action_mean, state_values = self.policy(old_obs)
            cov_mat = torch.diag(self.action_var)
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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            epoch_losses.append(loss.mean().item())

        # Log average loss for this update
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.losses.append(avg_loss)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"Checkpoint loaded from {checkpoint_path}")


# --- 4. Memory Buffer and Training Loop ---


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def append(self, action, state, logprob, reward, is_terminal):

        action = torch.as_tensor(action, dtype=torch.float32)
        state_on_device = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in state.items()
        }

        self.actions.append(action)
        self.states.append(state_on_device)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


