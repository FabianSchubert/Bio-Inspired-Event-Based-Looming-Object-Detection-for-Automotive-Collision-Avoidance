import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(
    "https://raw.githubusercontent.com/FabianSchubert/mpl_style/main/custom_style.mplstyle"
)

result = pd.read_csv("sim_speed.csv")

result.columns = ["gpu", "model", "Sim Steps / s"]

sns.barplot(x="gpu", y="Sim Steps / s", hue="model", data=result)

plt.show()

print(result)
