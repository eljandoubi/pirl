import os

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
from pathlib import Path

import cv2
import numpy as np
import robosuite as suite
import torch
import tqdm

from ppo import PPO, TrainingConfig
from robotenv import get_env_infos


def video_render(config = TrainingConfig()):

    # =========================
    # Config
    # =========================

    img_size = config.img_size
    max_episode_steps = config.max_ep_len
    device_id = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fps = 20
    checkpoint_best_path = f"{config.checkpoint_dir}/PPO_{config.env_name}_best.pth"
    video_path = checkpoint_best_path.replace("ppo_checkpoints", "ppo_videos").replace(".pth", ".mp4")
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    camera_names = (
        "frontview",
        "birdview",
        "agentview",
        "sideview",
        "robot0_robotview",
        "robot0_eye_in_hand",
    )
    keys = ["robot0_eye_in_hand_image", "robot0_eye_in_hand_depth", "robot0_proprio-state"]
    action_dim, obs_shapes = get_env_infos(img_size, keys)
    agent =  PPO(action_dim, device, obs_shapes, config)
    agent.load(checkpoint_best_path)
    # =========================
    # Agent policy 
    # =========================
    def to_torch(obs):
        result = {}
        for k in obs_shapes.keys():
            if "robot0_eye_in_hand" in k:
                obs[k] = cv2.resize(obs[k], (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                if "depth" in k:
                    obs[k] = obs[k][..., None]  # add channel dim for depth images


            result[k] = torch.as_tensor(obs[k], device=device, dtype=torch.float32).unsqueeze(0)
        return result

    @torch.inference_mode()
    def policy(obs):
        action, *_ = agent.select_action(to_torch(obs))
        action = action.squeeze(0).cpu().numpy()
        return action.clip(-1, 1)

    # =========================
    # Create environment
    # =========================
    env = suite.make(
        "Lift",
        robots="Panda",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=camera_names,
        camera_heights=img_size*4,
        camera_widths=img_size*4,
        reward_shaping=True,
        control_freq=20,
        horizon=max_episode_steps,
        render_gpu_device_id=device_id,
        camera_depths=True,
    )

    # =========================
    # Helpers
    # =========================
    def add_label(img, text):
        return cv2.putText(
            img.copy(),
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (127, 0, 127),
            2,
            cv2.LINE_AA,
        )

    def process_image(img, name):
        img = np.flipud(img)

        img = add_label(img, name)

        return img

    def compose_frame(obs):
        images = []

        for cam in camera_names:
            img = obs[f"{cam}_image"]
            img = process_image(img, cam)
            images.append(img)

        # layout: 3 rows × 2 columns
        row1 = np.concatenate(images[0:2], axis=1)  # front | bird
        row2 = np.concatenate(images[2:4], axis=1)  # agent | side
        row3 = np.concatenate(images[4:6], axis=1)  # robotview | eye_in_hand

        frame = np.concatenate([row1, row2, row3], axis=0)

        return frame


    # =========================
    # Run episode + collect frames
    # =========================
    frames = []

    obs = env.reset()
    cumulative_reward = 0
    for step in tqdm.tqdm(range(max_episode_steps), desc="Collecting frames"):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward

        frame = compose_frame(obs)

        # add global info
        cv2.putText(
            frame,
            f"Step: {step}  Reward: {reward:.2f}  Cumulative: {cumulative_reward:.2f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # convert RGB -> BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frames.append(frame)

        if done:
            break

    env.close()

    # =========================
    # Save video
    # =========================
    h, w, _ = frames[0].shape

    video = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for f in frames:
        video.write(f)

    video.release()

    print(f"✅ Video saved to {video_path}")
