import os

import torch

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #

import numpy as np
import robosuite as suite
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


def filter_state(state:dict[str, np.ndarray], keys:list[str])->dict[str, torch.Tensor]:
    return {
        k: torch.as_tensor(state[k], dtype=torch.float32) for k in keys
    }

def stack_obs(obs_list:list[dict[str, torch.Tensor]])->dict[str, torch.Tensor]:
    stacked_obs = {}
    for k in obs_list[0].keys():
        stacked_obs[k] = torch.stack([obs[k] for obs in obs_list], dim=0)
    return stacked_obs



def worker(remote, env_fn, env_kwargs, filter_keys):

    env = env_fn(**env_kwargs)

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            action = data
            obs_raw, reward, done, _ = env.step(action)

            if done:
                obs_raw = env.reset()

            obs = filter_state(obs_raw, filter_keys)

            remote.send((obs, reward, done))

        elif cmd == "reset":
            obs_raw = env.reset()
            obs = filter_state(obs_raw, filter_keys)
            remote.send(obs)

        elif cmd == "close":
            env.close()
            remote.close()
            break

class SubprocVecEnv:
    def __init__(self, env_fn, num_envs, env_kwargs, filter_keys):
        self.num_envs = num_envs
        self.filter_keys = filter_keys

        self.remotes, self.work_remotes = zip(*[
            mp.Pipe() for _ in range(num_envs)
        ])

        self.ps = [
            mp.Process(
                target=worker,
                args=(work_remote, env_fn, env_kwargs, filter_keys),
            )
            for work_remote in self.work_remotes
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))

        obs = [r.recv() for r in self.remotes]
        return stack_obs(obs)

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))

        results = [r.recv() for r in self.remotes]
        
        obs, rewards, dones = zip(*results)
        return stack_obs(obs), rewards, dones


    def close(self):
        for r in self.remotes:
            r.send(("close", None))

        for p in self.ps:
            p.join()