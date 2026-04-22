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
    try:
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
                break

    except Exception:
        # worker dies safely
        try:
            remote.close()
        except Exception:
            pass




class SubprocVecEnv:
    def __init__(self, env_fn, num_envs, env_kwargs, filter_keys, timeout=60):
        self.num_envs = num_envs
        self.env_fn = env_fn
        self.env_kwargs = env_kwargs
        self.filter_keys = filter_keys
        self.timeout = timeout

        self.remotes = [None] * num_envs
        self.work_remotes = [None] * num_envs
        self.ps = [None] * num_envs
        self.last_obs = [None] * num_envs

        for i in range(num_envs):
            self._spawn(i)

    # -------------------------
    # spawn (clean slot replace)
    # -------------------------
    def _spawn(self, i):
        parent, child = mp.Pipe()

        p = mp.Process(
            target=worker,
            args=(child, self.env_fn, self.env_kwargs, self.filter_keys),
        )
        p.daemon = True
        p.start()

        self.remotes[i] = parent
        self.work_remotes[i] = child
        self.ps[i] = p

        # immediate reset (guarantee valid obs)
        try:
            parent.send(("reset", None))
            self.last_obs[i] = parent.recv()
        except Exception:
            self.last_obs[i] = None

    # -------------------------
    # rebuild (atomic + safe)
    # -------------------------
    def _rebuild_worker(self, i):
        try:
            if self.ps[i] is not None:
                self.ps[i].terminate()
        except Exception:
            pass

        try:
            if self.remotes[i] is not None:
                self.remotes[i].close()
        except Exception:
            pass

        # clean replace
        self._spawn(i)

        return self.last_obs[i]

    # -------------------------
    # safe recv
    # -------------------------
    def _safe_recv(self, i):
        r = self.remotes[i]
        p = self.ps[i]

        if p is None or not p.is_alive():
            return None

        if not r.poll(self.timeout):
            return None

        try:
            return r.recv()
        except Exception:
            return None

    # -------------------------
    # reset (fixed-size always)
    # -------------------------
    def reset(self):
        obs = []

        for i in range(self.num_envs):
            try:
                self.remotes[i].send(("reset", None))
                o = self.remotes[i].recv()
                self.last_obs[i] = o
            except Exception:
                o = self._rebuild_worker(i)

            obs.append(o)

        return stack_obs(obs)

    # -------------------------
    # step (PATTERN 2 CORRECT)
    # -------------------------
    def step(self, actions):
        # send actions
        for i, (r, a) in enumerate(zip(self.remotes, actions)):
            try:
                r.send(("step", a))
            except Exception:
                # broken pipe → will rebuild on recv
                pass

        obs, rewards, dones = [], [], []

        for i in range(self.num_envs):
            result = self._safe_recv(i)

            # 🔥 HOT SWAP (single source of truth)
            if result is None:
                o = self._rebuild_worker(i)
                r = 0.0
                d = True
            else:
                o, r, d = result
                self.last_obs[i] = o

            obs.append(o)
            rewards.append(r)
            dones.append(d)

        return stack_obs(obs), rewards, dones

    # -------------------------
    # close
    # -------------------------
    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass

        for p in self.ps:
            try:
                if p is not None:
                    p.terminate()
                    p.join(timeout=1)
            except Exception:
                pass