import os

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
import time

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

        self.remotes = []
        self.work_remotes = []
        self.ps = []
        self.last_alive = []

        for i in range(num_envs):
            self._spawn(i)

        self.env_kwargs = env_kwargs
        self.last_obs = [None] * num_envs

    # -------------------------
    # spawn / respawn worker
    # -------------------------
    def _spawn(self, i):
        parent, child = mp.Pipe()

        p = mp.Process(
            target=worker,
            args=(child, self.env_fn, self.env_kwargs, self.filter_keys),
        )
        p.daemon = True
        p.start()

        self.remotes.append(parent)
        self.work_remotes.append(child)
        self.ps.append(p)
        self.last_alive.append(time.time())

    def _rebuild_worker(self, i):
        try:
            self.ps[i].terminate()
        except Exception:
            pass

        try:
            self.remotes[i].close()
        except Exception:
            pass

        self._spawn(i)

    # -------------------------
    # safe send/recv
    # -------------------------
    def _safe_recv(self, i):
        r = self.remotes[i]

        if not self.ps[i].is_alive():
            self._rebuild_worker(i)
            return None

        if not r.poll(self.timeout):
            # hung worker → restart
            self._rebuild_worker(i)
            return None

        try:
            return r.recv()
        except Exception:
            self._rebuild_worker(i)
            return None

    # -------------------------
    # reset
    # -------------------------
    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))

        obs = []
        for i in range(self.num_envs):
            o = self._safe_recv(i)
            if o is None:
                continue
            obs.append(o)

        return stack_obs(obs)

    # -------------------------
    # step
    # -------------------------
    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            try:
                r.send(("step", a))
            except Exception:
                # pipe broken → force respawn
                continue

        obs, rewards, dones = [], [], []

        for i in range(self.num_envs):
            r = self.remotes[i]

            # worker dead or stuck → hot swap immediately
            if (not self.ps[i].is_alive()) or (not r.poll(self.timeout)):
                self._rebuild_worker(i)

                # reset env after respawn
                self.remotes[i].send(("reset", None))
                o = self.remotes[i].recv()

                self.last_obs[i] = o
                obs.append(o)
                rewards.append(0.0)
                dones.append(True)
                continue

            try:
                o, rwd, done = r.recv()
            except Exception:
                # crash during recv → hot swap
                self._rebuild_worker(i)

                self.remotes[i].send(("reset", None))
                o = self.remotes[i].recv()

                self.last_obs[i] = o
                obs.append(o)
                rewards.append(0.0)
                dones.append(True)
                continue

            self.last_obs[i] = o
            obs.append(o)
            rewards.append(rwd)
            dones.append(done)

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
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass