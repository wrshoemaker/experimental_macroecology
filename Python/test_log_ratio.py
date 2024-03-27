import numpy
from scipy import stats

transfers=6
reps=10
x = stats.gamma.rvs(0.1, scale=3, size=(92, transfers))

print(x.shape)