# %%
import gym
import isaacgym
import isaacgymenvs
import torch
from isaacgym import gymapi, gymutil

num_envs = 20

envs = isaacgymenvs.make(
	seed=0, 
	task="Ant", 
	num_envs=20, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	headless=False,
	multi_gpu=False,
	virtual_screen_capture=False,
	force_render=False,
)

print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(20):
	obs, reward, done, info = envs.step(
		torch.rand((num_envs,)+envs.action_space.shape, device="cuda:0")
	)
	envs.render()

# %%

def set_camera(envs, position, lookat):
    """ Set camera position and direction
    """
    cam_pos = gymapi.Vec3(position[0], position[1], position[2])
    cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
    envs.gym.viewer_camera_look_at(envs.viewer, None, cam_pos, cam_target) \


# %%
set_camera(envs, [0, 0, 0], [0, 0, 0])

# %%
envs.render()

# %%

# def init_viewr(envs):
#     """ Initialize viewer
#     """
#     envs.viewer = envs.gym.create_viewer(
#         envs.sim, gymapi.CameraProperties())
#     envs.gym.subscribe_viewer_keyboard_event(
#         envs.viewer, gymapi.KEY_ESCAPE, "QUIT")
#     envs.gym.subscribe_viewer_keyboard_event(
#         envs.viewer, gymapi.KEY_V, "toggle_viewer_sync")


# %%
for _ in range(100):
	envs.step(
		torch.rand((20,)+envs.action_space.shape, device="cuda:0")
	)
	envs.render()