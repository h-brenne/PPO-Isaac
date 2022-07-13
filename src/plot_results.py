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
event_paths = glob.glob(os.path.join(path, "event*"))
df = tflogs2pandas.many_logs2pandas(event_paths)

print(df.head())
#df = tbconvert.tflog2pandas(path)
df = df[df["metric"] == "Metric/Score"]
sns.lineplot(data=df, x="step", y="value")
plt.show()