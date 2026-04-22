import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from model import ActorCritic


# --- 3. PPO Agent ---
class PPO:
    def __init__(
        self, action_dim, img_size, proprio_dim, lr_actor, lr_critic, gamma,
        K_epochs, eps_clip, device, fixed_policy_variance

    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        

        self.policy = ActorCritic(action_dim, img_size, proprio_dim,
                                  fixed_policy_variance=fixed_policy_variance).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(action_dim, img_size, proprio_dim,
                                      fixed_policy_variance=fixed_policy_variance).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # For logging losses
        self.losses = []

    def select_action(self, state):

        with torch.no_grad():
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
            cov_mat = torch.diag(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.detach().cpu(), action_logprob.detach().cpu()

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
        old_actions, old_states_img, old_states_depth, old_states_proprio, old_logprobs = memory.to_tensor()

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
    def __init__(self,device):
        self.device = device
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def append(self, action, state, logprob, reward, is_terminal):

        state_on_device = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in state.items()
        }

        self.actions.append(action)
        self.states.append(state_on_device)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def extend(self, actions, states, logprobs, rewards, is_terminals):
        states_on_device = []
        for state in states:
            state_on_device = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in state.items()
            }
            states_on_device.append(state_on_device)

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
    
    def to_tensor(self)->tuple[torch.Tensor,...]:
        actions = torch.stack(self.actions).to(self.device, non_blocking=True)
        states_img = torch.stack([s["agentview_image"] for s in self.states]).to(self.device, non_blocking=True)
        states_depth = torch.stack([s["agentview_depth"] for s in self.states]).to(self.device, non_blocking=True)
        states_proprio = torch.stack([s["robot0_proprio-state"] for s in self.states]).to(self.device, non_blocking=True)
        logprobs = torch.stack(self.logprobs).to(self.device, non_blocking=True)

        return actions, states_img, states_depth, states_proprio, logprobs
    
    def shape(self):
        res = {"len": len(self)}
        if len(self) > 0:
            res["actions"] = self.actions[0].shape
            res["states_img"] = self.states[0]["agentview_image"].shape
            res["states_depth"] = self.states[0]["agentview_depth"].shape
            res["states_proprio"] = self.states[0]["robot0_proprio-state"].shape
            res["logprobs"] = self.logprobs[0].shape
        return res


