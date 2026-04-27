import os  # noqa

# os.environ["MUJOCO_GL"] = "osmesa"  # "egl" #
import logging

import numpy as np
import robosuite as suite
import torch
import torch.multiprocessing as mp


def make_env(
    device_id: int = -1,
    img_size: int = 64,
    max_episode_steps: int = 200,
    reward_shaping: bool = True,
    use_object_obs: bool = False,
):

    env = suite.make(
        "Lift",
        robots="Panda",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names="robot0_eye_in_hand",
        camera_heights=img_size,
        camera_widths=img_size,
        camera_depths=True,
        reward_shaping=reward_shaping,
        horizon=max_episode_steps,
        render_gpu_device_id=device_id,
        use_object_obs=use_object_obs,
    )
    return env


def get_env_infos(img_size: int, keys: list[str], use_object_obs: bool = False):
    _env = make_env(img_size=img_size, use_object_obs=use_object_obs)
    action_dim = _env.action_dim
    obs_shapes = {k: _env.observation_spec()[k].shape for k in keys}
    _env.close()
    return action_dim, obs_shapes


def filter_state(
    state: dict[str, np.ndarray], keys: list[str]
) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(state[k], dtype=torch.float32) for k in keys}


def stack_obs(obs_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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


# -------------------------
# logger
# -------------------------
logger = logging.getLogger("SubprocVecEnv")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


class SubprocVecEnv:
    def __init__(self, env_fn, num_envs, env_kwargs, filter_keys, timeout=30):
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
    # fallback obs (CRITICAL)
    # -------------------------
    def _get_fallback_obs(self, i):
        if self.last_obs[i] is not None:
            return self.last_obs[i]

        logger.warning(f"[env {i}] trying fallback recovery")

        for _ in range(3):
            self._spawn(i)
            if self.last_obs[i] is not None:
                return self.last_obs[i]

        raise RuntimeError(f"[env {i}] cannot recover valid observation")

    # -------------------------
    # spawn worker
    # -------------------------
    def _spawn(self, i):
        logger.info(f"[env {i}] spawning worker")

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

        # initial reset (safe)
        try:
            parent.send(("reset", None))
            if parent.poll(self.timeout):
                self.last_obs[i] = parent.recv()
                logger.info(f"[env {i}] spawn + reset OK")
            else:
                logger.warning(f"[env {i}] spawn reset timeout")
                self.last_obs[i] = None
        except Exception as e:
            logger.error(f"[env {i}] spawn failed: {e}")
            self.last_obs[i] = None

    # -------------------------
    # rebuild worker
    # -------------------------
    def _rebuild_worker(self, i):
        logger.warning(f"[env {i}] rebuilding worker")

        try:
            if self.ps[i] is not None:
                self.ps[i].terminate()
        except Exception as e:
            logger.error(f"[env {i}] terminate failed: {e}")

        try:
            if self.remotes[i] is not None:
                self.remotes[i].close()
        except Exception as e:
            logger.error(f"[env {i}] close pipe failed: {e}")

        self._spawn(i)
        return self.last_obs[i]

    # -------------------------
    # safe recv (NON-BLOCKING)
    # -------------------------
    def _safe_recv(self, i):
        r = self.remotes[i]
        p = self.ps[i]

        if p is None or not p.is_alive():
            logger.warning(f"[env {i}] process dead")
            return None

        if not r.poll(self.timeout):
            logger.warning(f"[env {i}] recv timeout ({self.timeout}s)")
            return None

        try:
            return r.recv()
        except Exception as e:
            logger.error(f"[env {i}] recv failed: {e}")
            return None

    # -------------------------
    # reset
    # -------------------------
    def reset(self):
        logger.info("resetting all envs")

        obs = []

        for i in range(self.num_envs):
            try:
                self.remotes[i].send(("reset", None))
            except Exception as e:
                logger.error(f"[env {i}] reset send failed: {e}")

        for i in range(self.num_envs):
            o = self._safe_recv(i)

            if o is None:
                logger.warning(f"[env {i}] reset failed → rebuild")
                o = self._rebuild_worker(i)

            if o is None:
                logger.error(f"[env {i}] fallback obs used")
                o = self._get_fallback_obs(i)

            self.last_obs[i] = o
            obs.append(o)

        return stack_obs(obs)

    # -------------------------
    # step
    # -------------------------
    def step(self, actions):
        # send actions
        for i, (r, a) in enumerate(zip(self.remotes, actions)):
            try:
                r.send(("step", a))
            except Exception as e:
                logger.error(f"[env {i}] step send failed: {e}")

        obs, rewards, dones = [], [], []

        for i in range(self.num_envs):
            result = self._safe_recv(i)

            if result is None:
                logger.warning(f"[env {i}] step failed → hot swap")
                o = self._rebuild_worker(i)

                if o is None:
                    logger.error(f"[env {i}] fallback obs used")
                    o = self._get_fallback_obs(i)

                r = 0.0
                d = True
            else:
                o, r, d = result
                self.last_obs[i] = o

            obs.append(o)
            rewards.append(r)
            dones.append(d)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        return stack_obs(obs), rewards, dones

    # -------------------------
    # close
    # -------------------------
    def close(self):
        logger.info("closing all envs")

        for i, r in enumerate(self.remotes):
            try:
                r.send(("close", None))
            except Exception as e:
                logger.error(f"[env {i}] close send failed: {e}")

        for i, p in enumerate(self.ps):
            try:
                if p is not None:
                    p.terminate()
                    p.join(timeout=1)
            except Exception as e:
                logger.error(f"[env {i}] terminate failed: {e}")


if __name__ == "__main__":
    env_kwargs = {
        "device_id": -1,
        "img_size": 64,
        "max_episode_steps": 200,
        "reward_shaping": True,
        "use_object_obs": True,
    }
    env = make_env(**env_kwargs)
    obs = env.reset()
    for k, v in obs.items():
        print(f"{k}: {v.shape}")
    env.close()
