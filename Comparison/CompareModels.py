import arviz as az
from Utils.Conf import DOUBLE_RESULTS_PATH
import pickle
import matplotlib.pyplot as plt

n=5

analyzed_features1 = "hitter_pf_rate"
analyzed_features2 = "poly_hitter_pf_rate"

with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features1 + "_" + str(n) + ".pkl", 'rb') as handle:
    idata1 = pickle.load(handle)


with open(DOUBLE_RESULTS_PATH + "idata_m3_" + analyzed_features2 + "_" + str(n) + ".pkl", 'rb') as handle:
    idata2 = pickle.load(handle)

compare = az.compare(
    {
        analyzed_features1: idata1,
        analyzed_features2: idata2,
    },
)
az.plot_compare(compare, figsize=(10, 4))
plt.show()