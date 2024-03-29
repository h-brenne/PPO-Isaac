import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import dependencies.tflogs2pandas as tflogs2pandas 

sns.set_theme(style="darkgrid")

sweep_path = "runs/BallBalance/anneal_lr_Sweep/"
sub_entries = os.scandir(sweep_path)

for entry in sub_entries:
    if entry.is_dir():
        event_paths = glob.glob(os.path.join(sweep_path + entry.name + "/tb", "event*"))
        df = tflogs2pandas.many_logs2pandas(event_paths)
        df = df[df["metric"] == "Score/episode_reward"]
        #print(df.head())
        label = entry.name
        #print(df["wall_time"].iat[0])
        #df["wall_time"] -= df["wall_time"].iat[0]
        sns.lineplot(data=df, x="wall_time", y="value", label=label)
#plt.legend(labels=["Default init","", "Orthogonal init"])
plt.title("Orthogonal initialization")
plt.show()