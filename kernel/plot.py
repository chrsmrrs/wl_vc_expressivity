import matplotlib

matplotlib.use("TKAgg")
# matplotlib.use("macOSX")
from matplotlib import pyplot as plt

import seaborn as sns
import pandas as pd

import numpy as np



# for using latex in plt it requires one installation:
# $ sudo apt install dvipng
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
plt.rc('text', usetex=True)

# csns.set_theme(style="white")

data = np.loadtxt("data_wl_f.csv", delimiter=",")




df = pd.DataFrame(data, columns=["Difference", "Margin", "p"])

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

g = sns.scatterplot(x="Margin", y="Difference", hue="p", data=df, palette=sns.color_palette("flare", as_cmap=True) )
g.set(xlabel="Margin $\lambda$", ylabel="Train - test accuracy [\%]", title="1-$\mathsf{WL}_\mathcal{F}$")

sns.move_legend(g, "upper right", title='$p$')

plt.savefig('line_plot.pdf', bbox_inches = 'tight')

plt.show()
