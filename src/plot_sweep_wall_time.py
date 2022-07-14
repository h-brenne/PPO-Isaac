import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import dependencies.tflogs2pandas as tflogs2pandas 

sns.set_theme(style="darkgrid")

sweep_path = "runs/Humanoid/anneal_lr_Sweep/"
sub_entries = os.scandir(sweep_path)

for entry in sub_entries:
    if entry.is_dir():
        event_paths = glob.glob(os.path.join(sweep_path + entry.name + "/tb", "event*"))
        df_group = pd.DataFrame()
        for path in event_paths:
            df = tflogs2pandas.tflog2pandas(path)
            df["wall_time"] -= df["wall_time"].iat[0]
            if df_group.shape[0] == 0:
                df_group = df
            else:
                df_group = df_group.append(df, ignore_index=True)
        df_group = df_group[df_group["metric"] == "Score/episode_reward"]
        label = entry.name
        sns.lineplot(data=df_group, x="step", y="value", label=label)
plt.title("Orthogonal initialization")
plt.show()