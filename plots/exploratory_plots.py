from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('./tcc.mplstyle')


def pairplot(X, fig_path=None):
    pair = sns.pairplot(
        X.sample(frac=0.1),
        diag_kind='kde',
        height=2.75,
        aspect=1.25,
    )

    # rastering all subplots
    for ax in pair.axes.flatten():
        if ax.collections:
            ax.collections[0].set_rasterized(True)

    if fig_path is not None:
        pair.savefig(fig_path / 'pairplot.pdf', dpi=300)

    plt.close()


def corrplot(X, features, fig_path=None):
    correlation = X.corr()

    heat = sns.heatmap(
        correlation,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap='Blues',
        xticklabels=features,
        yticklabels=features,
        annot_kws={'fontsize': 14},
    )
    heat.axes.grid(False)
    plt.setp(heat.axes.get_xticklabels(), rotation=35, ha='right')

    if fig_path is not None:
        plt.savefig(fig_path / 'corrplot.pdf')

    plt.close()


def violinplot(X, features, fig_path=None):
    violin = sns.violinplot(
        X,
        orient='h',
        inner='box',
        width=1,
        gap=0.1,
        linewidth=0.5,
        density_norm='count',
    )

    violin.axes.set_yticks(violin.axes.get_yticks(), features)
    plt.xlabel('Amplitude')

    if fig_path is not None:
        plt.savefig(fig_path / 'violinplot.pdf')

    plt.close()


if __name__ == '__main__':
    DATASET_PATH = Path('D:/dados_tcc/dataset')
    FIG_PATH = Path('G:/Meu Drive/TCC/TCC Texto/Figuras')

    X = pd.read_feather(
        DATASET_PATH / 'X_clean.feather', columns=[0, 11, 15, 20]
    )
    features = [
        'Área abaixo da curva',
        'Potência máxima de espectro',
        'Média absoluta da derivada',
        'Distância de pico-a-pico',
    ]
    X.columns = features

    # plots
    pairplot(X, FIG_PATH)
    corrplot(X, features, FIG_PATH)
    violinplot(X, features, FIG_PATH)
