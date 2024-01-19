import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc

import seaborn as sns
import pandas as pd

import numpy as np

# for using latex in plt it requires one installation:
# $ sudo apt install dvipng
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

sns.set_theme(style="white")



matrix = np.array(
    [[1, 5.6, 0], [2, 3.4, 0], [3, 5.1, 0], [1, 1.2, 0], [2, 3.2, 0], [3, 3, 0], [1, 5.4, 0], [2, 3.2, 0], [3, 2.1, 0],
     [1, 3.2, 1], [2, 4.0, 1], [3, 5.1, 1], [1, 6.1, 1], [2, 3.2, 1], [3, 4.1, 1], [1, 3.2, 1], [2, 1.2, 1],
     [3, 1.1, 1]])

train_a = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
test_a = np.array([99.9, 99.9, 99.8, 99.9, 99.9, 99.8, 99.9, 99.9, 99.8, 99.9])
diff_a = train_a - test_a
margin_a = np.array([0.03868943, 0.04037981, 0.04120046, 0.04793041, 0.03845745, 0.04491592,
            0.03886781, 0.04003597, 0.04296789, 0.03989155])
label_a = np.array([0]*10)

train_b = np.array([100., 100., 100., 100., 100., 100., 100., 100., 100., 100.])
test_b = np.array([98.6, 98.7, 98.6, 98., 98.7, 98.6, 98.5, 98.3, 98.4, 98.5])
diff_b = train_b - test_b
margin_b = np.array([0.00890177, 0.00890607, 0.00908908, 0.00917992, 0.00903042, 0.00896859,
            0.0090599, 0.00886044, 0.00885055, 0.00883621])
label_b = np.array([1]*10)

train_c = np.array([100., 100., 100., 100., 99.98765432,
           100., 100., 100., 100., 100.])
test_c = np.array([79.8, 79.7, 79., 79.8, 80., 80.4, 79.2, 80.4, 79.8, 79.7])
diff_c = train_c - test_c
margin_c = np.array([0.00226358, 0.00226699, 0.00226476, 0.00226639, 0.00226999, 0.00227191,
            0.00226908, 0.00225886, 0.00227254, 0.00227362])
label_c = np.array([2]*10)

train_d = np.array([100., 100., 100., 100., 100.,
           100., 100., 99.98765432, 99.98765432, 100.])
test_d = np.array([7.8, 6.6, 7.1, 7.2, 7.2, 6.9, 6.8, 7.3, 8.3, 6.2])
diff_d = train_d - test_d
margin_d = np.array([0.00231986, 0.00246821, 0.00247205, 0.00232263, 0.00231544, 0.002313,
            0.00232557, 0.00232172, 0.00232673, 0.00232582])
label_d = np.array([3]*10)

diffs = np.concatenate([diff_a,diff_b,diff_c,diff_d]).reshape([40,1])
margins = np.concatenate([margin_a,margin_b,margin_c,margin_d]).reshape([40,1])
labels = np.concatenate([label_a,label_b,label_c,label_d]).reshape([40,1])

matrix = np.concatenate([diffs,margins,labels], axis=1)

df = pd.DataFrame(matrix, columns=["Difference", "Margin", "p"])



g = sns.lineplot(x="Margin", y="Difference", hue="p", data=df)
g.set(xlabel="Margin $\lambda$", ylabel="Train - test accuracy", title="1-$\mathsf{WL}_\mathcal{F}$")

sns.move_legend(g, "upper left", title='Prob.')

plt.show()
