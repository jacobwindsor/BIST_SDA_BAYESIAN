import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
from pathlib import Path

x_data = np.loadtxt(Path.cwd() / "michel.out")
plt.hist(x_data)
plt.title("Michel distribution from experimental data")
figpath = Path.cwd() / "histogram.png"
print(f"Histogram saved to {figpath}")

plt.savefig(figpath)