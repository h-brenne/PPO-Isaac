import glob
import os
import pandas as pd

import dependencies.tflogs2pandas as tflogs2pandas

def preprocess_df(path):
    event_paths = glob.glob(os.path.join(path, "event*"))
    df = tflogs2pandas.many_logs2pandas(event_paths)

    df = df[df["metric"] == "Score/episode_reward"]
    df["wall_time"] -= df["wall_time"].iat[0]
    return df