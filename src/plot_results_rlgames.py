import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from dependencies.tflogs2pandas import tflog2pandas
import preprocess_tb_df

sns.set_theme(style="darkgrid")

path1 = "runs/Humanoid/HighScore/tb"
path2 = "rlgames_runs/Humanoid/summaries/events.out.tfevents.1652461469.haavard"

df1 = preprocess_tb_df.preprocess_df(path1)

df2 = tflog2pandas(path2)
df2 = df2[df2["metric"] == "rewards/step"]
df2["wall_time"] -= df2["wall_time"].iat[0]

ax1 = sns.lineplot(data=df1, x="wall_time", y="value", label="PPO-Isaac")
ax2 = sns.lineplot(data=df2, x="wall_time", y="value", label="RLGames")
ax1.set_xlabel("Time, seconds")
ax1.set_ylabel("Reward")
plt.title("Humanoid")
plt.show()