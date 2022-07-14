# PPO-Isaac

PPO implementation for Isaac Gym. The goal of this implementation is to have a contained file for the PPO implementation such that algorithmic experimentation can be easier done. Tested on Preview 3 and Preview 4. 

## Installation
Install Isaac Gym Preview. Install [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs). **Important:** Use torch 1.12. With torch 1.8, sampling from a multivariate normal distribution with GPU tensors is extremely slow, and will hog most of the compute time If you setup a conda environment with the supplied script in Isaac Gym, you will get torch 1.8.

## Usage
Use 'python train.py', which takes the following arguments: 'env_cfg_path, train_cfg_path, run_name, checkpoint_path'
If a checkpoint path is given, training will start with the weights in checkpoint_path
Example with Humanoid: 'python train.py cfg/env/Humanoid.yaml cfg/algo/Humanoid_train.yaml'

sweep_param.py can be used to either perform a parameter sweep over a selected parameter, or collect multiple runs at different seeds with the same parameters. 

## Dependencies
src/dependencies contains tflogs2pandas.py from [supermariopy](https://github.com/theRealSuperMario/supermariopy). A small change was made to extract wall_time.
