import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import dependencies.tflogs2pandas as tflogs2pandas 

sns.set_theme(style="darkgrid")

path = "runs/BallBalance/multirun2/tb/"
#path = "runs/BallBalance/
event_paths = glob.glob(os.path.join(path, "event*"))
df = tflogs2pandas.many_logs2pandas(event_paths)

df = df[df["metric"] == "Score/episode_reward"]
sns.lineplot(data=df, x="step", y="value")
plt.show()