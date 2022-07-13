import subprocess

#Trick to clean up Isaac Gym by having the process with gym object close
for i in range(5):
    subprocess.check_call(['python', 'train.py', "cfg/env/BallBalance.yaml", "cfg/algo/BallBalance_train.yaml", "multirun2"])