import os

os.environ["MUJOCO_GL"] = "osmesa" # "egl" #
import robosuite as suite


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