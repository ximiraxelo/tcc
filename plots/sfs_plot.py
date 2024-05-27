from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

SAVE_PATH = Path('D:/dados_tcc/files')
FIG_PATH = Path('G:/Meu Drive/TCC/TCC Texto/Figuras')

with open(SAVE_PATH / 'sfs_metrics.joblib', 'rb') as file:
    metrics = joblib.load(file)

# from mlxtend.plotting.plot_sequential_feature_selection
k_feature = sorted(metrics.keys())
r2 = [metrics[k]['avg_score'] for k in k_feature]

with plt.style.context('./tcc.mplstyle'):
    plt.plot(k_feature, r2, 'ko-', markersize=3, label='$R^2$')
    plt.xlabel('Quantidade de atributos')
    plt.ylabel(r'Coeficiente de determinação $\left(R^2\right)$')
    plt.xticks(np.arange(1, 62, 3))
    plt.xlim([0, 63])

    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig(FIG_PATH / 'sfs.pdf')
    plt.show()
