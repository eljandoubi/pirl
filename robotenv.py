import os

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
import robosuite as suite
import torch.multiprocessing as mp


# --- 1. Environment Setup ---
# Function to create the robosuite environment
def make_env(device_id=-1, img_size=64):
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
        horizon=200,
        render_gpu_device_id=device_id,
    )
    return env


def worker(remote, env_fn, env_kwargs):
    env = env_fn(**env_kwargs)

    while True:
        cmd, data = remote.recv()

        if cmd == "step":
            obs, reward, done, _ = env.step(data)
            if done:
                obs = env.reset()
            remote.send((obs, reward, done))

        elif cmd == "reset":
            remote.send(env.reset())

        elif cmd == "close":
            env.close()
            remote.close()
            break


class VecEnv:
    def __init__(self, env_fn, num_envs, **env_kwargs):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])

        self.ps = [
            mp.Process(target=worker, args=(work_remote, env_fn, env_kwargs))
            for work_remote in self.work_remotes
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))

        results = [r.recv() for r in self.remotes]

        next_states, rewards, dones = zip(*results)
        return next_states, rewards, dones

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.ps:
            p.join()