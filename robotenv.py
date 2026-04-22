import os

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
import numpy as np
import robosuite as suite
import torch
import torch.multiprocessing as mp


def make_env(device_id=-1, img_size=64, max_episode_steps=200):
    env = suite.make(
        "Lift",
        robots="Panda",
        use_camera_obs=True,
        has_renderer=False, 
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=img_size,
        camera_widths=img_size,
        camera_depths=True,
        reward_shaping=True,
        control_freq=20,
        horizon=max_episode_steps,
        render_gpu_device_id=device_id,
    )
    return env


def filter_state(state:dict[str, np.ndarray], keys:list[str])->dict[str, np.ndarray]:
    return {
        k: state[k] for k in keys
    }


def worker(i, buffer, env_fn, env_kwargs, filter_keys):

    env = env_fn(**env_kwargs)
    obs = filter_state(env.reset(), filter_keys)

    while True:
        buffer.ready[i] = 0

        action = buffer.actions[i].numpy().clip(-1, 1)
        obs_raw, reward, done, _ = env.step(action)

        if done:
            obs_raw = env.reset()

        obs = filter_state(obs_raw, filter_keys)

        for k in buffer.obs:
            buffer.obs[k][i].copy_(torch.as_tensor(obs[k]))

        buffer.rewards[i] = reward
        buffer.dones[i] = float(done)

        buffer.ready[i] = 1

class SharedBuffer:
    def __init__(self, N, obs_shapes, action_dim):
        self.N = N

        self.obs = {
            k: torch.zeros((N, *shape), dtype=torch.float32).share_memory_()
            for k, shape in obs_shapes.items()
        }

        self.rewards = torch.zeros(N, dtype=torch.float32).share_memory_()
        self.dones = torch.zeros(N, dtype=torch.bool).share_memory_()
        self.actions = torch.zeros((N, action_dim), dtype=torch.float32).share_memory_()

        self.ready = torch.zeros(N, dtype=torch.int32).share_memory_()
        self.closed = torch.zeros(N, dtype=torch.int32).share_memory_()


class SharedVecEnv:
    def __init__(self, env_fn, num_envs, obs_shapes, action_dim, filter_keys, **env_kwargs):
        self.N = num_envs
        self.buffer = SharedBuffer(num_envs, obs_shapes, action_dim)

        self.ps = []
        for i in range(num_envs):
            p = mp.Process(
                target=worker,
                args=(i, self.buffer, env_fn, env_kwargs, filter_keys),
            )
            p.daemon = True
            p.start()
            self.ps.append(p)

    def reset(self):
        # wait until first observations are ready
        while self.buffer.ready.sum().item() < self.N:
            pass
        self.buffer.ready.zero_()

        return {k: v.clone() for k, v in self.buffer.obs.items()}

    def step(self, actions):
        # torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.buffer.actions.copy_(actions)

        while self.buffer.ready.sum().item() < self.N:
            pass
        self.buffer.ready.zero_()

        obs = {k: v.clone() for k, v in self.buffer.obs.items()}
        rewards = self.buffer.rewards.clone()
        dones = self.buffer.dones.clone()

        return obs, rewards, dones
    

    def close(self):
        # signal all workers to stop
        self.buffer.closed.fill_(1)

        # wait for processes to exit
        for p in self.ps:
            p.join(timeout=5)

        # force kill if stuck
        for p in self.ps:
            if p.is_alive():
                print("⚠️ Forcing worker termination")
                p.terminate()