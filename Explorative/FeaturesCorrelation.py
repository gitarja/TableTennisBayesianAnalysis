import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
features = ["skill", "personal", "perception", "action", "impact"]

n = len(features)
correlation_mat =  np.zeros((n, n))


for i in range(len(features)):
    features_i = np.load("Results\\" + features[i] + "_pred.npy")
    for j in range(len(features)):
        features_j = np.load("Results\\" + features[j] + "_pred.npy")

        # print(features_i)
        res = stats.pearsonr(features_i, features_j)
        correlation_mat[i, j] = res.statistic



g = sns.clustermap(correlation_mat, annot=True, fmt=".1f", cmap=sns.color_palette("Blues", as_cmap=True), vmin=0, vmax=1
            )

labels = ["Skill", "Personal", "Eye-movement", "Action", "Impact"]
# plt.xticks([0, 1, 2, 3, 4], labels=labels,  rotation=90)
# plt.yticks([0, 1, 2, 3, 4], labels=labels,  rotation=0)
plt.show()
