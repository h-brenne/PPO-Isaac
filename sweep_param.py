import subprocess
import yaml

def run_batch(env_cfg_path, train_cfg_path, num_seeds, run_name = 'None', ):
    for i in range(num_seeds):
        #Trick to clean up Isaac Gym by having the process with gym object close
        subprocess.check_call(['python', 'train.py', env_cfg_path, train_cfg_path, run_name])

#Run training over swept params, optionally at multiple seeds
def sweep_param(env_cfg_path, train_cfg_path, env_or_train, param_name, param_values, num_seeds):
    if env_or_train == "env":
        cfg_path = env_cfg_path
        env_cfg_path = "tmp/cfg.yaml"
    elif env_or_train == "train":
        cfg_path = train_cfg_path
        train_cfg_path = "tmp/cfg.yaml"
    else:
        print("Must specify config as env or train, not ", env_or_train)
        return

    with open(cfg_path, 'r') as stream:
        cfg=yaml.safe_load(stream)
    for i, value in enumerate(param_values):
        cfg[param_name] = value
        with open('tmp/cfg.yaml', 'w') as stream:
            yaml.safe_dump(cfg, stream)
        run_name = param_name + "_Sweep/" + str(value)
        run_batch(env_cfg_path, train_cfg_path, num_seeds, run_name = run_name)
            

if __name__ == "__main__":
    env_cfg_path = "cfg/env/BallBalance.yaml"
    train_cfg_path = "cfg/algo/BallBalance_train.yaml"
    
    sweep_param(env_cfg_path, train_cfg_path, "train", "rollout_steps", [4, 16, 64], 2)
    #run_batch(env_cfg_path, train_cfg_path)