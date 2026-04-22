import torch


class RolloutBuffer:
    def __init__(self, T:int, N:int, obs_shapes:dict[str,tuple[int,...]],
                 action_dim:int, device:torch.device):
        """
        T: Number of time steps to store
        N: Number of parallel environments
        obs_shapes: Dictionary mapping observation keys to their shapes (excluding batch dimension)
        action_dim: Dimension of the action space
        device: PyTorch device to store the tensors on
        """
        self.T, self.N = T, N
        self.device = device

        self.actions = torch.zeros(T, N, action_dim, device=device, dtype=torch.float32)
        self.rewards = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.dones = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.values = torch.zeros(T, N, device=device, dtype=torch.float32)
        self.logprobs = torch.zeros(T, N, device=device, dtype=torch.float32)

        self.obs = {
            k: torch.zeros(T, N, *shape, device=device, dtype=torch.float32)
            for k, shape in obs_shapes.items()
        }
        self.obs_shapes = obs_shapes
        self.ptr = 0

    def __repr__(self):
        return f"RolloutBuffer(T={self.T}, N={self.N}, ptr={self.ptr}, device={self.device}, obs={self.obs_shapes})"

    def add(self, obs:dict[str, torch.Tensor], actions:torch.Tensor, rewards:torch.Tensor,
            dones:torch.Tensor, values:torch.Tensor, logprobs:torch.Tensor):
        t = self.ptr

        for k in self.obs:
            self.obs[k][t] = torch.as_tensor(obs[k], device=self.device, dtype=torch.float32)

        self.actions[t] = actions
        self.rewards[t] = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        self.dones[t] = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        self.values[t] = values.squeeze(1)
        self.logprobs[t] = logprobs

        self.ptr += 1
        assert self.ptr <= self.T, f"Buffer overflow: ptr {self.ptr} exceeds T {self.T}"

    @torch.no_grad()
    def compute_gae(self, last_value:torch.Tensor, gamma:float=0.99, lam:float=0.95):
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.N, device=self.device)

        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]

            mask = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]

            gae = delta + gamma * lam * mask * gae
            adv[t] = gae

        returns = adv + self.values
        return returns, adv

    def get(self):
        T, N = self.T, self.N

        batch = {"obs": {}}

        for k in self.obs:
            batch["obs"][k] = self.obs[k].reshape(T * N, *self.obs_shapes[k])

        batch["actions"] = self.actions.reshape(T * N, -1)
        batch["logprobs"] = self.logprobs.reshape(T * N)
        # batch["values"] = self.values.reshape(T * N)
        # batch["rewards"] = self.rewards.reshape(T * N)
        # batch["dones"] = self.dones.reshape(T * N)

        return batch

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr*self.N