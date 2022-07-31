# PPO-Isaac

PPO implementation for [Isaac Gym Benchmark Envs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs). The goal of this implementation is to have a contained file for the PPO implementation such that algorithmic experimentation can be easier done. Tested on Preview 3 and Preview 4. 

## Installation
Install [Isaac Gym Preview](https://developer.nvidia.com/isaac-gym). Install [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs). **Important:** Use torch 1.12. With torch 1.8, sampling from a multivariate normal distribution with GPU tensors is extremely slow, and will hog most of the compute time. If you setup a conda environment with the supplied script in Isaac Gym, you will get torch 1.8.

## Usage
Use 'python train.py', which takes the following arguments: 'env_cfg_path, train_cfg_path, run_name, checkpoint_path'
If a checkpoint path is given, training will start with the weights in checkpoint_path
Example with Humanoid: 'python train.py cfg/env/Humanoid.yaml cfg/algo/Humanoid_train.yaml'

sweep_param.py can be used to either perform a parameter sweep over a selected parameter, or collect multiple runs at different seeds with the same parameters. 

## Dependencies
src/dependencies contains tflogs2pandas.py from [supermariopy](https://github.com/theRealSuperMario/supermariopy). A small change was made to extract wall_time.

## Results
Trained on a GTX1070ti using hyperparameters found in cfg folder. Comparison is between the default RLGames implementation with default hyperparameters from [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
### AnymalTerrain
https://user-images.githubusercontent.com/12870693/181915045-89b78919-79af-446f-ba53-cf9c1c647ce2.mp4

![AnymalTerrainVS](https://user-images.githubusercontent.com/12870693/181915675-40beb184-beea-47f5-8d99-b0eb9ba5ff93.png)

### Humanoid
https://user-images.githubusercontent.com/12870693/181914889-5eb56265-da9a-4222-bc60-08767521e75d.mp4

![HuimanoidVS](https://user-images.githubusercontent.com/12870693/181915839-5318c057-7d9f-4fd5-8098-0075f55c28e3.png)

### Ant
https://user-images.githubusercontent.com/12870693/181914888-3abd64a7-a409-46b4-bd1a-a048a2ad32ad.mp4

![AntVS](https://user-images.githubusercontent.com/12870693/181915841-786db676-acd5-46a6-ae52-ccc9b1e024eb.png)

### BallBalance
https://user-images.githubusercontent.com/12870693/181914868-2e00f062-0319-4031-aa65-af3540394d04.mp4

![BallBalanceVS](https://user-images.githubusercontent.com/12870693/181915844-61b65de5-e841-48bd-b710-e831876226ac.png)


