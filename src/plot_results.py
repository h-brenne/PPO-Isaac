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

df1 = preprocess_tb_df.preprocess_df(path1)

ax1 = sns.lineplot(data=df1, x="wall_time", y="value", label="PPO-Isaac")
ax1.set_xlabel("Time, seconds")
ax1.set_ylabel("Reward")
plt.title("Humanoid")
plt.show()