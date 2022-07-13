from src.PPO import PPO
import sys

if __name__ == "__main__":
    env_cfg_path = "cfg/env/Cartpole.yaml"
    train_cfg_path = "cfg/algo/Cartpole_train.yaml"
    run_name = "None"
    checkpoint_path = "None"

    num_args = len(sys.argv)-1
    if num_args == 1:
        print("Expected 0 or minimum 2 arguments: env_cfg_path, train_cfg_path")
        quit()
    if num_args >= 2:
        env_cfg_path = sys.argv[1]
        train_cfg_path = sys.argv[2]
    if num_args == 3:
        run_name = sys.argv[3]
    elif num_args == 4:
        run_name = sys.argv[3]
        checkpoint_path = sys.argv[4]
    elif num_args > 4:
        print("Expected maximum 4 arguments: env_cfg_path, train_cfg_path, run_name, checkpoint_path")
        quit()
    
    print("Run name: ", run_name)
    learner = PPO(env_cfg_file = env_cfg_path, train_cfg_file = train_cfg_path, run_name = run_name, checkpoint = checkpoint_path)
    learner.run()
