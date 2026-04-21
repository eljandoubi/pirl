import os
from pathlib import Path

# Logging and plotting
import matplotlib.pyplot as plt
import robosuite as suite
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


# --- 1. Environment Setup ---
# Function to create the robosuite environment
def make_env():
    env = suite.make(
        "Lift",
        robots="Panda",
        use_camera_obs=True,
        has_renderer=False,  # Set to True to visualize
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        camera_depths=True,
        reward_shaping=True,
        control_freq=20,
        horizon=200,
    )
    return env


# --- 2. Multi-Modal Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()

        # Image processing network (CNN) for RGB
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Depth processing network (CNN) for Depth
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of the flattened CNN outputs
        # A dummy forward pass helps in determining the flat feature size
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 84, 84)
            dummy_depth = torch.zeros(1, 1, 84, 84)
            img_feature_size = self.image_conv(dummy_img).shape[1]
            depth_feature_size = self.depth_conv(dummy_depth).shape[1]

        # Proprioceptive state processing network (MLP)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(50, 128),  # Proprio state size for Panda is 50
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Fusion layer
        fused_size = img_feature_size + depth_feature_size + 64
        self.fusion_layer = nn.Sequential(nn.Linear(fused_size, 512), nn.ReLU())

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(512, action_dim),
            nn.Tanh(),  # To scale actions to [-1, 1]
        )

        # Critic head
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        img = obs["agentview_image"]
        depth = obs["agentview_depth"]
        proprio = obs["robot0_proprio-state"]

        # Permute image dimensions to be (batch, channel, height, width)
        img = img.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)

        img_features = self.image_conv(img)
        depth_features = self.depth_conv(depth)
        proprio_features = self.proprio_mlp(proprio)

        # Concatenate features
        fused_features = torch.cat(
            (img_features, depth_features, proprio_features), dim=1
        )
        fused_output = self.fusion_layer(fused_features)

        action_mean = self.actor(fused_output)
        state_value = self.critic(fused_output)

        return action_mean, state_value


# --- 3. PPO Agent ---
class PPO:
    def __init__(
        self, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.action_var = torch.full((action_dim,), 0.5**2).to(device)

        # For logging losses
        self.losses = []

    def select_action(self, state):
        with torch.no_grad():
            # Preprocess state for the network
            img = (
                torch.tensor(state["agentview_image"]).to(self.device).unsqueeze(0)
            )
            depth = (
                torch.tensor(state["agentview_depth"]).to(self.device).unsqueeze(0)
            )
            proprio = (
                torch.tensor(state["robot0_proprio-state"])
                .to(self.device)
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states_img = torch.tensor(
            [s["agentview_image"] for s in memory.states]
        ).to(self.device)
        old_states_depth = torch.tensor(
            [s["agentview_depth"] for s in memory.states]
        ).to(self.device)
        old_states_proprio = torch.tensor(
            [s["robot0_proprio-state"] for s in memory.states]
        ).to(self.device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).to(self.device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).to(
            self.device
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
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
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
    def __init__(self, device=None):
        self.device = device
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def append(self, action, state, logprob, reward, is_terminal):
        # Move action and logprob to device if not already
        if isinstance(action, torch.Tensor):
            action = action.to(self.device)
        if isinstance(logprob, torch.Tensor):
            logprob = logprob.to(self.device)
        # Move state arrays to device as tensors
        state_on_device = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
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


def main():

    ############## Hyperparameters ##############
    env_name = "robosuite_lift"
    max_ep_len = 200  # max timesteps in one episode
    max_training_timesteps = (
        1000000  # break training loop if timeteps > max_training_timesteps
    )

    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

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

    env = make_env()
    action_dim = env.action_spec[0].shape[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    memory = Memory(device=device)
    ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    if load_checkpoint_path:
        ppo_agent.load(load_checkpoint_path)

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action, log_prob = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals

            memory.append(torch.from_numpy(action), state, log_prob, reward, done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update(memory)
                memory.clear_memory()

            # save model checkpoint
            if time_step % save_model_freq == 0:
                checkpoint_path = f"{checkpoint_dir}/PPO_{env_name}_{time_step}.pth"
                ppo_agent.save(checkpoint_path)

            if done:
                break

        i_episode += 1

        # Logging reward
        reward_history.append(current_ep_reward)
        with open(reward_log_path, "a") as f:
            f.write(f"{current_ep_reward}\n")

        print(
            f"Episode {i_episode} \t Timestep: {time_step} \t Reward: {current_ep_reward}"
        )

    env.close()

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
