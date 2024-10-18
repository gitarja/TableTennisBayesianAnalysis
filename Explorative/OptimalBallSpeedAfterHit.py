import numpy as np
import pandas as pd
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH
import arviz as az
import matplotlib.pyplot as plt

df = pd.read_pickle(DOUBLE_SUMMARY_FEATURES_PATH)

df = df.loc[df["success"] == 1]
df = df.loc[df["team_skill"] > 0.65]

data = df[[
    "ball_speed_after_hit",
    "ball_dir_after_hit"]
].values

print(np.median(data[:, 0]))
print(np.median(data[:, 1]))
az.plot_kde(data[:, 0])
plt.show()
az.plot_kde(data[:, 1])
plt.show()
