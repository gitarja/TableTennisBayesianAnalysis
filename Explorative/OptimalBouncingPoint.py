import numpy as np
import pandas as pd
from Utils.Conf import DOUBLE_SUMMARY_FEATURES_PATH
import arviz as az
import matplotlib.pyplot as plt
df = pd.read_pickle(DOUBLE_SUMMARY_FEATURES_PATH)

df = df.loc[df["success"] == 1]
df = df.loc[df["team_skill"] > 0.65]

bouncing_point_p1 = df[[
    "bouncing_point_p1_x",
    "bouncing_point_p1_z",
]].values



bouncing_point_p2 = df[[

    "bouncing_point_p2_x",
    "bouncing_point_p2_y"]].values

print(np.median(bouncing_point_p1, axis=0))
print(np.median(bouncing_point_p2, axis=0))

ax = az.plot_kde(
    bouncing_point_p1[:, 0],
    bouncing_point_p1[:, 1],


    contourf_kwargs={"cmap": "Blues"},
)


plt.show()
ax = az.plot_kde(
    bouncing_point_p2[:, 0],
    bouncing_point_p2[:, 1],


    contourf_kwargs={"cmap": "Blues"},
)


plt.show()