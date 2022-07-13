import subprocess

#Trick to clean up Isaac Gym by having the process with gym object close
for i in range(10):
    subprocess.check_call(['python', 'train.py', "cfg/env/Cartpole.yaml", "cfg/algo/Cartpole_train.yaml", "multirun"])