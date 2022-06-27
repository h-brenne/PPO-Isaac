from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np

import torch

device="cuda:0"

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = True

#We want to use z-up 
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)



# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "../assets"
asset_file = "urdf/bicopter.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)

asset_options = gymapi.AssetOptions()
#asset_options.fix_base_link = True
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
num_dof = gym.get_asset_dof_count(cartpole_asset)

pose = gymapi.Transform()
pose.p.z = 2.0
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    actor_handle = gym.create_actor(env, cartpole_asset, pose, "cart", i, 1)

    """dof_props = gym.get_actor_dof_properties(env, actor_handle)
    dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
    dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
    dof_props['stiffness'][:] = 0.0
    dof_props['damping'][:] = 0.0
    gym.set_actor_dof_properties(env, actor_handle, dof_props)"""

    actor_handles.append(actor_handle)

#Run graphics:
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

#Tensor API to prepare for CPU or GPU
gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):

    ###Pre physic step
    max_force = 100
    #forces_torch = max_force*(1.0-2.0 * torch.rand((num_envs*3, 3), dtype=torch.float32, device=device))

    forces_torch = torch.tensor([[0, 0, 20],[0,0,0],[0,0,0]], dtype=torch.float32, device=device)
    forces_torch = forces_torch.repeat(num_envs,1)
    #Torch tensor to raw tensor
    forces = gymtorch.unwrap_tensor(forces_torch)

    #pos = torch.tensor([0,0.5,0], dtype=torch.float32, device=device)
    #pos = pos.repeat(num_envs*3,1)
    #pos = gymtorch.unwrap_tensor(pos)

    #gym.apply_rigid_body_force_at_pos_tensors(sim, forces, pos, gymapi.LOCAL_SPACE)
    gym.apply_rigid_body_force_tensors(sim, forces, None, gymapi.LOCAL_SPACE)
    
    
    
    #gym.set_dof_actuation_force_tensor(sim, forces)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)