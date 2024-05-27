from sklearnex import patch_sklearn  # noreorder

patch_sklearn()
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

plt.style.use('./tcc.mplstyle')

DATASET_PATH = Path('D:/dados_tcc/dataset')
FIG_PATH = Path('G:/Meu Drive/TCC/TCC Texto/Figuras')

X4 = pd.read_feather(DATASET_PATH / 'X_clean.feather', columns=[0, 11, 15, 20])
y = np.load(DATASET_PATH / 'y_clean.npy')

X4_train, X4_test, y_train, y_test = train_test_split(
    X4, y, test_size=0.2, random_state=123
)

X4_train, X4_test, y_train, y_test = train_test_split(
    X4_train, y_train, test_size=0.2, random_state=321
)

scaler = RobustScaler()
X4_train = scaler.fit_transform(X4_train)
X4_test = scaler.transform(X4_test)
X3_train = X4_train[:, 0:3]
X3_test = X4_test[:, 0:3]

labels = [
    'SVM',
    'Rede Neural',
    'XGBoost',
    'Árvore de Decisão',
    'Floresta Aleatória',
    'KNN',
]

models = [
    LinearSVR(
        random_state=123,
        dual=False,
        loss='squared_epsilon_insensitive',
        max_iter=5000,
    ),
    MLPRegressor(
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=123,
    ),
    XGBRegressor(random_state=123),
    DecisionTreeRegressor(max_depth=13, random_state=123),
    RandomForestRegressor(
        max_depth=13,
        random_state=123,
    ),
    KNeighborsRegressor(),
]

score3, score4 = [], []

for label, model in zip(labels, models):
    print(f'\n[purple]---- {label} ----\n')

    print('[green]3 atributos', end=' ')
    model.fit(X3_train, y_train)
    r2 = model.score(X3_test, y_test)
    score3.append(r2)
    print(f'[green]-> {r2}')

    print('[green]4 atributos', end=' ')
    model.fit(X4_train, y_train)
    r2 = model.score(X4_test, y_test)
    score4.append(r2)
    print(f'[green]-> {r2}')

x1 = np.arange(len(labels))
x2 = x1 + 0.4

plt.figure(figsize=(10, 5))
plt.bar(x1, score3, width=0.4, label='3 Atributos')
plt.bar(x2, score4, width=0.4, label='4 Atributos')
plt.grid(False, axis='x')
plt.xlabel('Modelos')
plt.xticks(x1 + 0.2, labels, fontsize=12)
plt.ylabel(r'Coeficiente de determinação $\left(R^2\right)$')
plt.ylim((0.3, 1))
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH / 'features_barplot.pdf')
plt.show()
